import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.s2s.transformer import Transformer
from models.losses import MLCrossEntropy
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

def evaluate_loss(model, dataloader, w):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                image_id, samples, labels, tokens_kd = batch['image_id'], batch['samples'], batch['labels'], batch['tokens_kd']
                samples['grid'] = samples['grid'].to(device)
                samples['mask'] = samples['mask'].to(device)
                labels = labels.to(device)
                tokens_kd = tokens_kd.to(device)
                en_out, out = model(samples, labels)

                # optim.zero_grad()
                # XE
                seq_len = min(out.shape[1], tokens_kd.shape[1])
                out_ce = out[:, :seq_len].contiguous()
                tokens_kd = tokens_kd[:, :seq_len].contiguous()
                loss_ce = loss_fn(out_ce.view(-1, len(text_field.vocab)), tokens_kd.view(-1))

                # ML
                # loss_ml = loss_fn(en_out.view(-1, en_out.shape[-1]), labels.view(-1))
                loss = loss_ce # * w + loss_ml
                # loss.backward()

                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1), loss_ce=loss_ce.item()) #, loss_ml=loss_ml.item())
                pbar.update()

                if test:
                    break

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, caps_gt, labels = batch['image_id'], batch['samples'], batch['caps_gt'], batch['labels']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _, logit = model(samples, labels)
            
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

def train_xe(model, dataloader, optim, text_field, w):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, labels, tokens_kd = batch['image_id'], batch['samples'], batch['labels'], batch['tokens_kd']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            labels = labels.to(device)
            tokens_kd = tokens_kd.to(device)
            en_out, out = model(samples, labels)
            
            optim.zero_grad()
            # XE
            seq_len = min(out.shape[1], tokens_kd.shape[1])
            out_ce = out[:, :seq_len].contiguous()
            tokens_kd = tokens_kd[:, :seq_len].contiguous()
            loss_ce = loss_fn(out_ce.view(-1, len(text_field.vocab)), tokens_kd.view(-1))

            # ML
            # loss_ml = loss_fn(en_out.view(-1, en_out.shape[-1]), labels.view(-1))

            loss = loss_ce # * w + loss_ml
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1), loss_ce=loss_ce.item()) #, loss_ml=loss_ml.item())
            pbar.update()

            if test:
                break
    
    loss = running_loss / len(dataloader)
    return loss

def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    beam_size = 1
    seq_len = 20

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, caps_gt = batch['image_id'], batch['samples'], batch['caps_gt']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            _, logit = model(samples)
            batch_size = logit.shape[0]
            max_len = logit.shape[1]
            optim.zero_grad()

            sents_logprobs, sents = torch.topk(logit, 2)
        
            sents_copy = sents[:,:,:1].squeeze(-1)
            caps_gen = text_field.decode(sents_copy.view(-1, sents_copy.shape[-1]), deduplication=True)
            caps_gt1 = caps_gt
            caps_gen, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt1])
            rewards_sample = cider.compute_score(caps_gt1, caps_gen)[1].astype(np.float32)
            rewards_sample = torch.from_numpy(rewards_sample).to(device).reshape(batch_size, 1).repeat(1, seq_len)

            sents_0 = sents[:,:,:1].squeeze(-1).unsqueeze(1).repeat(1, seq_len, 1)
            sents_1 = sents[:,:,1:].squeeze(-1).unsqueeze(1).repeat(1, seq_len, 1)
            x_1 = torch.eye(seq_len).to(device).unsqueeze(0).unsqueeze(0)
            x_0 = torch.where(x_1==1, 0, 1)
            sents_re = sents_0*x_0 + sents_1*x_1
            caps_re = text_field.decode(sents_re.view(-1, sents_re.shape[-1]), deduplication=True)
            caps_gt2 = list(itertools.chain(*([c, ] * seq_len for c in caps_gt)))
            caps_re, caps_gt2 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_re, caps_gt2])
            rewards_re = cider.compute_score(caps_gt2, caps_re)[1].astype(np.float32)
            rewards_re = torch.from_numpy(rewards_re).to(device).reshape(batch_size, seq_len)

            sents_logprobs_0 = sents_logprobs[:,:,:1].squeeze(-1)
            sents_logprobs_1 = sents_logprobs[:,:,1:].squeeze(-1)
            reward_baseline = (sents_logprobs_0*rewards_sample + sents_logprobs_1*rewards_re)/(sents_logprobs_0+sents_logprobs_1)
            loss = - sents_logprobs_0 * (rewards_sample - reward_baseline)
            loss = loss.mean()

            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += rewards_sample.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
            if test:
                break

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

def train_scst1(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    beam_size = 1
    seq_len = 20

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, caps_gt = batch['image_id'], batch['samples'], batch['caps_gt']
            samples['grid'] = samples['grid'].to(device)
            samples['mask'] = samples['mask'].to(device)
            _, logit = model(samples)
            batch_size = logit.shape[0]
            max_len = logit.shape[1]
            optim.zero_grad()


            logit_e = torch.exp(logit)
            h = -torch.sum(logit_e * logit, -1)
            _, id_re = torch.max(h, -1)

            sents_logprobs, sents = torch.topk(logit, 2)
        
            sents_copy = sents[:,:,:1].squeeze(-1)
            caps_gen = text_field.decode(sents_copy.view(-1, sents_copy.shape[-1]), deduplication=True)
            caps_gt1 = caps_gt
            caps_gen, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt1])
            rewards_sample = cider.compute_score(caps_gt1, caps_gen)[1].astype(np.float32)
            rewards_sample = torch.from_numpy(rewards_sample).to(device)

            sents_0 = sents[:,:,:1].squeeze(-1)
            sents_1 = sents[:,:,1:].squeeze(-1)
            x_1 = F.one_hot(id_re, num_classes=seq_len)
            x_0 = torch.where(x_1==1, 0, 1)
            sents_re = sents_0*x_0 + sents_1*x_1
            caps_re = text_field.decode(sents_re.view(-1, sents_re.shape[-1]), deduplication=True)
            caps_gt2 = caps_gt
            caps_re, caps_gt2 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_re, caps_gt2])
            rewards_re = cider.compute_score(caps_gt2, caps_re)[1].astype(np.float32)
            rewards_re = torch.from_numpy(rewards_re).to(device)

            sents_logprobs_0 = torch.gather(sents_logprobs[:,:,:1].squeeze(-1), dim=1, index=id_re.unsqueeze(0)).squeeze()
            sents_logprobs_1 = torch.gather(sents_logprobs[:,:,1:].squeeze(-1), dim=1, index=id_re.unsqueeze(0)).squeeze()
            reward_baseline = (sents_logprobs_0*rewards_sample + sents_logprobs_1*rewards_re)/(sents_logprobs_0+sents_logprobs_1)
            loss = - sents_logprobs_0 * (rewards_sample - reward_baseline)
            loss = loss.mean()

            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += rewards_sample.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
            if test:
                break

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    device = torch.device('cuda')
    args = OmegaConf.load('configs/s2s.yaml')
    print(args)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.mode, args.exp_name))

    dataloaders, text_field = build_coco_dataloaders(args, device)
    cider_train = Cider()

    if args.dataset.use_cache:
        detector = None
    else:
        detector = build_detector(args)
    
    model = Transformer(len(text_field.vocab), text_field.vocab.stoi['<pad>']).to(device)

    for n, p in model.named_parameters():
        if 'detector' in n:
            p.requires_grad = False

    def lambda_lr(s):
        base_lr = 0.0001
        if s <= 2:
            lr = base_lr * (s+1) / 4
        elif s <= 25:
            lr = base_lr
        elif s <= 30:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.5
        print("Epoch: %d, Learning Rate: %f" % (s, lr))
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_fn_1 = MLCrossEntropy()
    best_cider = .0
    patience = 0
    start_epoch = 0
    use_rl = False

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
            scheduler.step()
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        w = e+1
        if not use_rl:
            train_loss = train_xe(model, dataloaders['train'], optim, text_field, w)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dataloaders['train'], optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        if not use_rl:
            # Validation loss
            val_loss = evaluate_loss(model, dataloaders['valid'], w)
            writer.add_scalar('data/val_loss', val_loss, e)
        else:
            val_loss = 0.0

        # Validation scores
        scores = evaluate_metrics(model, dataloaders['valid'], text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dataloaders['test'], text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if test_cider >= best_cider:
            best_cider = test_cider
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
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
        }, os.path.join(args.model_path, '%s_last.pth' % args.mode))

        if best:
            copyfile(os.path.join(args.model_path, '%s_last.pth' % args.mode), os.path.join(args.model_path, '%s_best.pth' % args.mode))
        if exit_train:
            writer.close()
            break
        
        scheduler.step()