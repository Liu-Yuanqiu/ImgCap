import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer_word import Transformer
from models.losses import FocalLossWithLogitsNegLoss
from models.metric import MultiLabelAccuracy, mAPMeter
from pycocotools.coco import COCO
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from omegaconf import OmegaConf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def evaluate_metrics(model, dataloader, topk):
    import itertools
    model.eval()
    running_acc = 0
    acc = 0
    with tqdm(desc='Evaluation - %d' % topk, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model(samples)
            
            out = out.sigmoid()
            topk_ids = out.topk(k=topk, dim=1)[1]
            res = torch.zeros_like(out)
            for i in range(res.shape[0]):
                res[i].scatter_(dim=0, index=topk_ids[i], src=torch.ones_like(res[i]))
            target = (labels>0).float()
            this_acc = (res*target).sum(dim=1) / res.sum(dim=1)
            
            running_acc += this_acc.mean().item()
            acc = running_acc / (it+1)

            pbar.set_postfix(acc=acc)
            pbar.update()
    return acc

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    device = torch.device('cuda')
    args = OmegaConf.load('configs/s2s_word.yaml')
    print(args)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.mode, args.exp_name))

    dataloaders, text_field = build_coco_dataloaders(args, device)
    # if test:
    #     sys.exit()
    
    model = Transformer(len(text_field.vocab), text_field.vocab.stoi['<pad>']).to(device)

    # for n, p in model.named_parameters():
    #     if 'detector' in n:
    #         p.requires_grad = False

    def lambda_lr(s):
        base_lr = 0.0001
        if s <= 2:
            lr = base_lr * (s+1) / 4
        elif s <= 7:
            lr = base_lr
        elif s <= 10:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        print("Epoch: %d, Learning Rate: %f" % (s, lr))
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    
    loss_fn = FocalLossWithLogitsNegLoss()
    best_acc = 0.0
    patience = 0
    start_epoch = 0

    args.model_path = os.path.join("./ckpts", args.mode, args.exp_name)
    fname = os.path.join(args.model_path, '%s_best.pth' % args.mode)

    if os.path.exists(fname):
        data = torch.load(fname)
        model.load_state_dict(data['state_dict'], strict=False)
        print('Resuming from epoch %d, validation loss %f' % (
                data['epoch'], data['val_loss']))

    for topk in [5, 10, 20]:
        # Validation scores
        acc = evaluate_metrics(model, dataloaders['valid'], topk)
        # Test scores
        acc = evaluate_metrics(model, dataloaders['test'], topk)