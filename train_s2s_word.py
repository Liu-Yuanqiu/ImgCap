import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer_word import Transformer
from models.losses import FocalLossWithLogitsNegLoss, WeightedFocalLossWithLogitsNegLoss
from models.metric import MultiLabelAccuracy, mAPMeter
from pycocotools.coco import COCO
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.nn import NLLLoss, MSELoss
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

def evaluate_loss(model, dataloader):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
                samples['grid'] = samples['grid'].to(device)
                samples['mask'] = samples['mask'].to(device)
                labels = labels.to(device)
                en_out = model(samples)

                loss_fl = loss_fn0(en_out, labels)
                loss_fl = loss_fl.mean()
                loss =  loss_fl
                # loss.backward()

                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1), loss_fl=loss_fl.item())
                pbar.update()

                if test:
                    break

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader):
    model.eval()
    running_acc = 0
    acc = 0
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model(samples)
            
            out = out.sigmoid()
            topk_ids = out.topk(k=20, dim=1)[1]
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

def train_xe(model, dataloader, optim):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            
            image_id, samples, labels = batch['image_id'], batch['samples'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            en_out = model(samples)
            
            optim.zero_grad()

            loss_fl = loss_fn0(en_out, labels)
            loss_fl = loss_fl.mean()
            loss =  loss_fl
            loss.backward()

            optim.step()
            
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1), loss_fl=loss_fl.item())
            pbar.update()

            if test:
                if it%100==0:
                    print("Epoch: %d, it: %d, Learning Rate: %f" % (e, it, optim.param_groups[0]['lr']))
                break
    
    loss = running_loss / len(dataloader)
    return loss

if __name__ == '__main__':
    args = OmegaConf.load('configs/s2s_word.yaml')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')

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
        elif s <= 15:
            lr = base_lr
        elif s <= 30:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        print("Epoch: %d, Learning Rate: %f" % (s, lr))
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=args.optimizer.lr, betas=(0.9, 0.98))
    # scheduler = LambdaLR(optim, lambda_lr)
    # scheduler = CosineAnnealingLR(optim, T_max=args.optimizer.t_max, eta_min=args.optimizer.min_lr)
    # scheduler = ExponentialLR(optimizer=optim, gamma=args.optimizer.gamma)
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=3)
    
    loss_fn0 = WeightedFocalLossWithLogitsNegLoss()
    # loss_fn1 = MSELoss()
    best_acc = 0.0
    patience = 0
    start_epoch = 0

    args.model_path = os.path.join("./ckpts", args.mode, args.exp_name)
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.model_path, '%s_last.pth' % args.mode)
        else:
            fname = os.path.join(args.model_path, '%s_best.pth' % args.mode)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            # scheduler.step()
            start_epoch = data['epoch'] + 1
            # best_cider = data['best_cider']
            patience = data['patience']
            print('Resuming from epoch %d, validation loss %f' % (
                data['epoch'], data['val_loss']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        print("Epoch: %d, Learning Rate: %f" % (e, optim.param_groups[0]['lr']))
        train_loss = train_xe(model, dataloaders['train'], optim)
        writer.add_scalar('data/train_loss', train_loss, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloaders['valid'])
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        val_acc = evaluate_metrics(model, dataloaders['valid'])
        writer.add_scalar('data/val_acc', val_acc, e)

        # Test scores
        test_acc = evaluate_metrics(model, dataloaders['test'])
        writer.add_scalar('data/test_acc', test_acc, e)

        # Prepare for next epoch
        best = False
        if test_acc >= best_acc:
            best_acc = test_acc
            patience = 0
            best = True
        else:
            patience += 1

        exit_train = False
        if patience == 10:
            print('patience reached.')
            exit_train = True

        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_acc': best_acc,
        }, os.path.join(args.model_path, '%s_last.pth' % args.mode))

        if best:
            copyfile(os.path.join(args.model_path, '%s_last.pth' % args.mode), os.path.join(args.model_path, '%s_best.pth' % args.mode))
        if exit_train:
            writer.close()
            break
        
        scheduler.step(test_acc)