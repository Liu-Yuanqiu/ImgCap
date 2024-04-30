import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer_word import Transformer
from models.losses import FocalLossWithLogitsNegLoss
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

def evaluate_metrics(model, dataloader):
    import itertools
    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model(samples)
            
            out = out.sigmoid()
            out = torch.where(out > 0.4, torch.ones_like(out), torch.zeros_like(out))
            # _, pred_ids = torch.topk(out, 20, dim=-1)
            # pred = torch.zeros_like(out)
            # pred1 = torch.ones_like(out)
            # pred.scatter_(1, pred_ids, pred1)

            tptn = (out == labels).float().sum()
            tp_now = (out * labels).sum()
            tn_now = tptn - tp_now
            fn_now = ((out-labels) == 1).float().sum()
            fp_now = ((labels-out) == 1).float().sum()
            tp += tp_now
            tn += tn_now
            fp += fp_now
            fn += fn_now

            pbar.update()
    pre_avg = tp / (tp + fp)
    rec_avg = tp / (tp + fn)
    print('All Precision: %f, Recall: %f, F1: %f' % (pre_avg, rec_avg, 2*pre_avg*rec_avg/(pre_avg+rec_avg) ))
    return pre_avg, rec_avg, 2*pre_avg*rec_avg/(pre_avg+rec_avg)

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
    # if args.resume_last or args.resume_best:
        # if args.resume_last:
    fname = os.path.join(args.model_path, '%s_last.pth' % args.mode)
        # else:
            # fname = os.path.join(args.model_path, '%s_best.pth' % args.mode)

    if os.path.exists(fname):
        data = torch.load(fname)
        torch.set_rng_state(data['torch_rng_state'])
        torch.cuda.set_rng_state(data['cuda_rng_state'])
        np.random.set_state(data['numpy_rng_state'])
        random.setstate(data['random_rng_state'])
        model.load_state_dict(data['state_dict'], strict=False)
        optim.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        scheduler.step()
        start_epoch = data['epoch'] + 1
        # best_cider = data['best_cider']
        patience = data['patience']
        print('Resuming from epoch %d, validation loss %f' % (
                data['epoch'], data['val_loss']))
        print('patience:', data['patience'])


    # Validation scores
    acc_avg, rec_avg, f1 = evaluate_metrics(model, dataloaders['valid'])

    # Test scores
    acc_avg, rec_avg, f1 = evaluate_metrics(model, dataloaders['test'])