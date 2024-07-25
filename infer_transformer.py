import random
import evaluation
from evaluation import Cider
from data.dataset import build_coco_dataloaders
from models.detector import build_detector
from models.transformer import TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Transformer
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from omegaconf import OmegaConf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def infer(model, dataloader, text_field):
    model.eval()
    data = []
    with tqdm(desc='infer', unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, captions = batch['image_id'], batch['samples'], batch['captions']
            samples['grid'] = samples['grid'].to(device)
            with torch.no_grad():
                out, _ = model.beam_search(samples['grid'], 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=True)
            caps_gt = captions
            for i, (id, gts_i, gen_i) in enumerate(zip(image_id, caps_gt, caps_gen)):
                d = {}
                d['image_id'] = id
                d['gen'] = gen_i
                d['gts'] = gts_i
                data.append(d)
            pbar.update()
    return data

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    device = torch.device('cuda')
    args = OmegaConf.load('configs/transformer.yaml')
    print(args)

    dataloaders, text_field = build_coco_dataloaders(args, device)

    # Model and dataloaders
    if args.exp_mode == 'transformer':
        if args.dataset.use_cache:
            detector = None
        else:
            detector = build_detector(args)
        encoder = TransformerEncoder(3, text_field.vocab.stoi['<pad>'])
        decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    args.model_path = os.path.join("./ckpts", args.exp_mode, args.exp_name)
    fname = os.path.join(args.model_path, '%s_best.pth' % args.exp_name)
    assert os.path.exists(fname), "weight is not found"
    data = torch.load(fname)
    torch.set_rng_state(data['torch_rng_state'])
    torch.cuda.set_rng_state(data['cuda_rng_state'])
    np.random.set_state(data['numpy_rng_state'])
    random.setstate(data['random_rng_state'])
    model.load_state_dict(data['state_dict'], strict=False)
    print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    data = []
    for tag in ["train_dict", "valid_dict", "test_dict"]:
        data += infer(model, dataloaders[tag], text_field)
    
    with open(os.path.join(args.dataset.root_path, "annotations", "captions_"+args.exp_name+".json"), "w") as f:
        json.dump(data, f)