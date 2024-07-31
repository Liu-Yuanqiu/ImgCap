import random
import evaluation
from evaluation import Cider
from data.dataset_kd_h5 import build_coco_dataloaders
from models.naicdm.naicdm import Transformer
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import argparse
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
                label, token_kd = batch['label'], batch['token_kd']
                label = label.to(device)
                token_kd = token_kd.to(device)
                if args.origin_fea == "swin_dert_grid":
                    feat, mask = batch["feat"].to(device), batch["mask"].to(device)
                elif args.origin_fea == "swin_dert_region":
                    feat, mask = batch["feat"].to(device), batch["mask"].to(device)
                elif args.origin_fea == "up_down_36":
                    feat, mask = batch["feat"].to(device), None
                else:
                    raise NotImplementedError
                
                losses = model(feat, mask, label, token_kd)

                loss = 0
                for v in losses.values():
                    loss += v
                this_loss = loss.item()
                running_loss += this_loss
                losses_info = {}
                for k in losses:
                    item = '%.4f' % losses[k].item()
                    losses_info[k] = item
                pbar.set_postfix(loss=running_loss / (it + 1), losses=losses_info)
                pbar.update()

                if args.test:
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
            cap_gt = batch['cap_gt']
            if args.origin_fea == "swin_dert_grid":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif args.origin_fea == "swin_dert_region":
                feat, mask = batch["feat"].to(device), batch["mask"].to(device)
            elif args.origin_fea == "up_down_36":
                feat, mask = batch["feat"].to(device), None
            else:
                raise NotImplementedError
            
            with torch.no_grad():
                logit = model.infer(feat, mask)
            
            _, out = torch.max(logit, -1)
            caps_gen = text_field.decode(out, join_words=False, deduplication=True)
            for i, (gts_i, gen_i) in enumerate(zip(cap_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

            if args.test:
                break

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    loop = args.loop
    running_loss = .0
    torch.autograd.set_detect_anomaly(True)
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)*loop) as pbar:
        for i in range(loop):
            for it, batch in enumerate(dataloader):
                label, token_kd = batch['label'], batch['token_kd']
                
                label = label.to(device)
                token_kd = token_kd.to(device)
                if args.origin_fea == "swin_dert_grid":
                    feat, mask = batch["feat"].to(device), batch["mask"].to(device)
                elif args.origin_fea == "swin_dert_region":
                    feat, mask = batch["feat"].to(device), batch["mask"].to(device)
                elif args.origin_fea == "up_down_36":
                    feat, mask = batch["feat"].to(device), None
                else:
                    raise NotImplementedError
            
                losses = model(feat, mask, label, token_kd)
                # print(losses)
                optim.zero_grad()
                loss = 0
                for v in losses.values():
                    loss += v
                loss.backward()

                optim.step()
                this_loss = loss.item()
                running_loss += this_loss
                losses_info = {}
                for k in losses:
                    item = '%.4f' % losses[k].item()
                    losses_info[k] = item
                pbar.set_postfix(loss=running_loss / (i*len(dataloader) + it + 1), losses=losses_info)
                pbar.update()

                if args.test and it==10:
                    break
            if args.test:
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
            if args.test:
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
    parser = argparse.ArgumentParser(description='NAICDM')
    parser.add_argument('--rank', type=str, default='0')
    parser.add_argument('--exp_mode', type=str, default='naicdm')
    parser.add_argument('--exp_name', type=str, default='step3_up_down_36_topk5_step10')
    parser.add_argument('--log_folder', type=str, default='./logs')
    parser.add_argument('--data_path', type=str, default='../mscoco')
    parser.add_argument('--origin_cap', type=str, default='up_down_36')
    parser.add_argument('--origin_fea', type=str, default='up_down_36')
    parser.add_argument('--use_cache', action='store_true', default='True')
    parser.add_argument('--resume_step2', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num_timesteps', type=int, default=10)
    parser.add_argument('--sample_timesteps', type=int, default=10)
    parser.add_argument('--loop', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epoch1', type=int, default=100)
    parser.add_argument('--epoch2', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    if args.test:
        args.batch_size = 4
        args.workers = 2
        args.exp_name = "test"
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')

    writer = SummaryWriter(log_dir=os.path.join(args.log_folder, args.exp_mode, args.exp_name))

    dataloaders, text_field, stop_words = build_coco_dataloaders(args.data_path, args.batch_size, args.workers, origin_cap=args.origin_cap)
    cider_train = Cider()
    print("UNK: %d, PAD: %d, BOS: %d, EOS: %d" % (text_field.vocab.stoi['<unk>'], text_field.vocab.stoi['<pad>'], text_field.vocab.stoi['<bos>'], text_field.vocab.stoi['<eos>']))
    if args.origin_fea == "swin_dert_grid":
        args.feat_dim = 1024
    elif args.origin_fea == "swin_dert_region":
        args.feat_dim = 512
    elif args.origin_fea == "up_down_36":
        args.feat_dim = 2048
    else:
        raise NotImplementedError
    model = Transformer(args.feat_dim, len(text_field.vocab), text_field.vocab.stoi['<pad>'], args.topk, \
                        args.num_timesteps, args.sample_timesteps,\
                        N_en=args.layer_num, N_wo=args.layer_num, N_de=args.layer_num).to(device)
    model.tensor_to(device)
    def lambda_lr(s):
        base_lr = args.learning_rate
        if s <= 3:
            lr = base_lr * (s+1) / 4
        elif s <= args.epoch1:
            lr = base_lr
        elif s <= args.epoch2:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr
    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    best_cider = .0
    start_epoch = 0
    patience = 0
    use_rl = False

    if args.resume_step2:
        step2_model = "step2_" + args.origin_fea + "_topk" + str(args.topk)
        fname = os.path.join("./ckpts", args.exp_mode, step2_model, '%s_best.pth' % args.exp_mode)
        data = torch.load(fname)
        torch.set_rng_state(data['torch_rng_state'])
        torch.cuda.set_rng_state(data['cuda_rng_state'])
        np.random.set_state(data['numpy_rng_state'])
        random.setstate(data['random_rng_state'])
        model.load_state_dict(data['state_dict'], strict=False)
        print("Loaded step2 model from %s, best cider %f" % (fname, data['best_cider']))

    args.model_path = os.path.join("./ckpts", args.exp_mode, args.exp_name)
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.model_path, '%s_last.pth' % args.exp_mode)
        else:
            fname = os.path.join(args.model_path, '%s_best.pth' % args.exp_mode)

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
            print('Resuming from epoch %d, patience %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['patience'], data['val_loss'], data['best_cider']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        print("Epoch: %d, Learning Rate: %f" % (e, optim.param_groups[0]['lr']))
        if not use_rl:
            train_loss = train_xe(model, dataloaders['train'], optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dataloaders['train'], optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
            
        if not use_rl:
            # Validation loss
            val_loss = evaluate_loss(model, dataloaders['valid'])
            writer.add_scalar('data/val_loss', val_loss, e)
        else:
            val_loss = 0.0

        # Validation scores
        scores = evaluate_metrics(model, dataloaders['val_test'], text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dataloaders['test_test'], text_field)
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
        if patience == args.patience:
            if args.epoch1==100:
                args.epoch1 = e
                patience = 0
            else:
                if args.epoch2==200:
                    args.epoch2 = e
                    patience = 0
                else:
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
        }, os.path.join(args.model_path, '%s_last.pth' % args.exp_mode))

        if best:
            copyfile(os.path.join(args.model_path, '%s_last.pth' % args.exp_mode), os.path.join(args.model_path, '%s_best.pth' % args.exp_mode))
        if exit_train:
            writer.close()
            break
        
        scheduler.step()
