import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer import Transformer
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def evaluate_metrics(model, dataloader, topk):
    import itertools
    model.eval()
    running_len = 0
    running_acc = 0
    acc = 0
    with tqdm(desc='Evaluation - %f' % topk, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model.forward_word(samples)
            
            out = out.sigmoid()
            res = torch.zeros_like(out)
            if topk>1:
                topk_ids = out.topk(k=topk, dim=1)[1]
                for i in range(res.shape[0]):
                    res[i].scatter_(dim=0, index=topk_ids[i], src=torch.ones_like(res[i]))
            else:
                outt = out.gt(topk)
                for i,t in enumerate(outt):
                    topk_ids = torch.nonzero(t, as_tuple=False).squeeze(1)
                    res[i].scatter_(dim=0, index=topk_ids, src=torch.ones_like(res[i]))

            target = (labels>0).float()
            this_acc = (res*target).sum(dim=1) / (res.sum(dim=1) + 1e-10)
            
            running_len += res.sum(dim=1).mean().item()
            running_acc += this_acc.mean().item()
            # acc = running_acc / (it+1)

            pbar.set_postfix(acc=running_acc / (it+1), len=running_len / (it+1))
            pbar.update()
    return acc

if __name__ == '__main__':
    args = OmegaConf.load('configs/s2sw.yaml')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_mode, args.exp_name))

    dataloaders, text_field = build_coco_dataloaders(args, device)
    
    model = Transformer(len(text_field.vocab), text_field.vocab.stoi['<pad>'], args.topk).to(device)

    args.model_path = os.path.join("./ckpts", args.exp_mode, args.exp_name)
    fname = os.path.join(args.model_path, '%s_best.pth' % args.exp_mode)
    if os.path.exists(fname):
        data = torch.load(fname)
        model.load_state_dict(data['state_dict'], strict=False)
        print('Resuming from epoch %d, validation loss %f' % ( data['epoch'], data['val_loss']))

    for topk in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 5, 10, 20]:
        # Validation scores
        acc = evaluate_metrics(model, dataloaders['valid'], topk)
        # Test scores
        acc = evaluate_metrics(model, dataloaders['test'], topk)