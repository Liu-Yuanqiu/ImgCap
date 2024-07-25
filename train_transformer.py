import random
import evaluation
from evaluation import Cider
from data.dataset import build_coco_dataloaders
from models.detector import build_detector
from models.transformer import TransformerEncoder, TransformerDecoder, Transformer

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


test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def evaluate_loss(model, dataloader, loss_fn, text_field):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                image_id, samples, captions = batch['image_id'], batch['samples'], batch['captions']
                samples['grid'] = samples['grid'].to(device)
                if samples['mask'] is None:
                    samples['mask'] = torch.zeros(samples['grid'].shape[:2], dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(1)
                else:
                    samples['mask'] = samples['mask'].to(device)
                captions = captions.to(device)
                out = model(samples['grid'], captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
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
            image_id, samples, captions = batch['image_id'], batch['samples'], batch['captions']
            samples['grid'] = samples['grid'].to(device)
            if samples['mask'] is None:
                samples['mask'] = torch.zeros(samples['grid'].shape[:2], dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(1)
            else:
                samples['mask'] = samples['mask'].to(device)
            # captions = captions.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(samples['grid'], 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gt = captions
            
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, captions = batch['image_id'], batch['samples'], batch['captions']
            samples['grid'] = samples['grid'].to(device)
            if samples['mask'] is None:
                samples['mask'] = torch.zeros(samples['grid'].shape[:2], dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(1)
            else:
                samples['mask'] = samples['mask'].to(device)
            captions = captions.to(device)
            out = model(samples['grid'], captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
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
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, batch in enumerate(dataloader):
            image_id, samples, captions = batch['image_id'], batch['samples'], batch['captions']
            samples['grid'] = samples['grid'].to(device)
            if samples['mask'] is None:
                samples['mask'] = torch.zeros(samples['grid'].shape[:2], dtype=torch.bool, device=device).unsqueeze(1).unsqueeze(1)
            else:
                samples['mask'] = samples['mask'].to(device)
            # captions = captions.to(device)
            outs, log_probs = model.beam_search(samples['grid'], seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = captions
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(outs.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
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
    args = OmegaConf.load('configs/transformer.yaml')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    multiprocessing.set_start_method('spawn')
    
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_mode, args.exp_name))

    dataloaders, text_field = build_coco_dataloaders(args, device)
    cider_train = Cider()

    # Model and dataloaders
    if args.exp_mode == 'transformer':
        if args.dataset.use_cache:
            detector = None
        else:
            detector = build_detector(args)
        encoder = TransformerEncoder(3, text_field.vocab.stoi['<pad>'])
        decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
        model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    for n, p in model.named_parameters():
        if 'detector' in n:
            p.requires_grad = False

    def lambda_lr(s):
        base_lr = 0.0001
        # if s == 0:
            # lr = base_lr / 4
        if s < 3:
            lr = base_lr * (s+1) / 4
        elif s <= 5:
            lr = base_lr
        elif s <= 11:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr

    def lambda_lr_rl(s):
        base_lr = 5e-6
        print("s:", s)
        if s <= 100:
            lr = base_lr
        elif s <= 200:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    args.model_path = os.path.join("./ckpts", args.exp_mode, args.exp_name)
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = os.path.join(args.model_path, '%s_last.pth' % args.exp_name)
        else:
            fname = os.path.join(args.model_path, '%s_best.pth' % args.exp_name)

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
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        print("Epoch: %d, Learning Rate: %f" % (e, optim.param_groups[0]['lr']))
        if not use_rl:
            train_loss = train_xe(model, dataloaders['train'], optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dataloaders['train_dict'], optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
        scheduler.step()
        # Validation loss
        val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dataloaders['valid_dict'], text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dataloaders['test_dict'], text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if args.epoch1==100:
                args.epoch1 = e
                patience = 0
            else:
                if args.epoch2==200:
                    args.epoch2 = e
                    patience = 0
                else:
                    if not use_rl:
                        use_rl = True
                        switch_to_rl = True
                        patience = 0
                        optim = Adam(model.parameters(), lr=5e-6)
                        print("Switching to RL")
                        xe_path = os.path.join("./ckpts", args.exp_mode, args.exp_name+"_xe")
                        if not os.path.isdir(xe_path):
                            os.makedirs(xe_path)
                        copyfile(os.path.join(args.model_path, '%s_best.pth' % args.exp_name), os.path.join(xe_path, '%s_best.pth' % args.exp_name))
                        copyfile(os.path.join(args.model_path, '%s_last.pth' % args.exp_name), os.path.join(xe_path, '%s_last.pth' % args.exp_name))
                    else:
                        print('patience reached.')
                        exit_train = True
        #####
        # if not use_rl:
        #     if e >= 20:
        #         use_rl = True
        #         switch_to_rl = True
        #         patience = 0
        #         optim = Adam(model.parameters(), lr=5e-6)
        #         print("Switching to RL")
        #######

        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        if switch_to_rl and not best:
            data = torch.load(os.path.join(args.model_path, '%s_best.pth' % args.exp_name))
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

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
            'use_rl': use_rl,
        }, os.path.join(args.model_path, '%s_last.pth' % args.exp_name))

        if best:
            copyfile(os.path.join(args.model_path, '%s_last.pth' % args.exp_name), os.path.join(args.model_path, '%s_best.pth' % args.exp_name))
        if exit_train:
            writer.close()
            break
