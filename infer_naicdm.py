import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders, build_coco_dataloaders_test4w
from models.naicdm.naicdm_wordemb import Transformer
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
import time
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

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            samples, labels, caps_gt = batch['samples'], batch['labels'], batch['caps_gt']
            if origin_fea == "swin_dert_grid":
                feat, mask = samples["grid_sd"].to(device), samples["grid_sd_mask"].to(device)
            elif origin_fea == "swin_dert_region":
                feat, mask = samples["region_sd"].to(device), samples["region_sd_mask"].to(device)
            elif origin_fea == "up_down_36":
                feat, mask = samples["region_ud"].to(device), None
            else:
                raise NotImplementedError
            with torch.no_grad():
                logit = model.infer(feat, mask)
            
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=False, deduplication=True)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')
    origin_cap = "up_down_36"
    origin_fea = "up_down_36"
    if origin_fea == "swin_dert_grid":
        feat_dim = 1024
    elif origin_fea == "swin_dert_region":
        feat_dim = 512
    elif origin_fea == "up_down_36":
        feat_dim = 2048
    else:
        raise NotImplementedError
    step = 100
    root_path = "../mscoco"
    loaders, text_field, stop_words = build_coco_dataloaders(root_path, 64, 4, origin_cap)
    loader4w, load3w, text_field = build_coco_dataloaders_test4w()

    # Model and dataloaders
    model = Transformer(feat_dim, len(text_field.vocab), text_field.vocab.stoi['<pad>'], \
                        20, num_timesteps=step, sampling_timesteps=step,\
                        N_en=3, N_wo=3, N_de=3).to(device)
    model.tensor_to(device)
    model_path = os.path.join("./ckpts", "naicdm", "up_down_36")
    fname = os.path.join(model_path, '%s_best.pth' % "naicdm")
    assert os.path.exists(fname), "weight is not found"
    data = torch.load(fname)
    # torch.set_rng_state(data['torch_rng_state'])
    # torch.cuda.set_rng_state(data['cuda_rng_state'])
    # np.random.set_state(data['numpy_rng_state'])
    # random.setstate(data['random_rng_state'])
    model.load_state_dict(data['state_dict'], strict=False)
    print('Resuming from epoch %d, best cider %f' % (
                data['epoch'], data['best_cider']))
    model.eval()
    scores = evaluate_metrics(model, loaders["val_test"], text_field)
    print("Validation scores", scores)
    scores = evaluate_metrics(model, loaders["test_test"], text_field)
    print("Test scores", scores)
    
    data4w = []
    with tqdm(desc='infer', unit='it', total=len(loader4w)) as pbar:
        for it, batch in enumerate(loader4w):
            start = time.time()
            image_id, samples = batch['image_id'], batch['samples']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            with torch.no_grad():
                logit = model.infer(samples)
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=True, deduplication=True)
            for i, (id, gen_i) in enumerate(zip(image_id, caps_gen)):
                d = {}
                d['image_id'] = id
                d['caption'] = gen_i
                data4w.append(d)
            pbar.update()
    all_time = time.time() - start
    print("per image: %f" % (all_time/len(loader4w)))
    with open("./captions_test2014_results.json", "w") as f:
        json.dump(data4w, f)

    data3w = []
    with tqdm(desc='infer', unit='it', total=len(load3w)) as pbar:
        for it, batch in enumerate(load3w):
            start = time.time()
            image_id, samples = batch['image_id'], batch['samples']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            with torch.no_grad():
                logit = model.infer(samples)
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=True, deduplication=True)
            for i, (id, gen_i) in enumerate(zip(image_id, caps_gen)):
                d = {}
                d['image_id'] = id
                d['caption'] = gen_i
                data3w.append(d)
            pbar.update()
    all_time = time.time() - start
    print("per image: %f" % (all_time/len(load3w)))
    with open("./captions_val2014_results.json", "w") as f:
        json.dump(data3w, f)