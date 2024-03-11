import random
import evaluation
from evaluation import Cider
from data.dataset_kd import build_coco_dataloaders
from models.detector import build_detector
from models.mlnaic import TransformerEncoder, TransformerDecoder, Transformer
from models.losses import MLCrossEntropy
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
test = False
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def beam_search(logit, seq_len, beam_size):
    bs = logit.shape[0]
    now_prob, now_cap = torch.topk(logit[:, 0].squeeze(), beam_size, dim=-1)
    all_prob = now_prob
    now_prob = now_prob.unsqueeze(-1)
    now_cap = now_cap.unsqueeze(-1)
    
    for i in range(1, seq_len, 1):
        now_logit = logit[:, i:i+1].squeeze()
        # [bs, bm]
        i_prob, i_word = torch.topk(now_logit, beam_size, dim=-1)
        
        i_now_cap = now_cap.repeat(1, beam_size, 1)
        i_now_cap = torch.cat((i_now_cap, i_word.unsqueeze(-1).repeat(1, 1, beam_size).view(bs, beam_size*beam_size, 1)), -1)
        i_now_prob = now_prob.repeat(1, beam_size, 1)
        i_now_prob = torch.cat((i_now_prob, i_prob.unsqueeze(-1).repeat(1, 1, beam_size).view(bs, beam_size*beam_size, 1)), -1)
        i_all_prob = all_prob.repeat(1, beam_size)
        i_all_prob = i_all_prob * i_prob.unsqueeze(-1).repeat(1, 1, beam_size).view(bs, beam_size*beam_size)

        top_probs, top_ids = torch.topk(i_all_prob, beam_size, dim=-1)
        # top_probs, top_ids = torch.sort(i_all_prob, beam_size, dim=-1)
        now_cap = torch.gather(i_now_cap, 1, top_ids.unsqueeze(-1).expand(bs, beam_size, i_now_cap.shape[-1]))
        now_prob = torch.gather(i_now_prob, 1, top_ids.unsqueeze(-1).expand(bs, beam_size, i_now_prob.shape[-1]))
        all_prob = torch.gather(i_all_prob, 1, top_ids)
    return now_prob.contiguous(), now_cap.contiguous()

def evaluate_loss(model, dataloader, loss_fn, text_field):

    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                image_id, samples, labels, label_masks = batch['image_id'], batch['samples'], batch['labels'], batch['label_masks']

                out = model(samples)
                optim.zero_grad()
                min_len = min(out.shape[1], labels.shape[1])
                labels = labels[:, :min_len].contiguous()
                out = out[:, :min_len].contiguous()
                label_masks = label_masks[:, :min_len].contiguous()

                loss = loss_fn(out, labels, label_masks)
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
            image_id, samples, caps_gt = batch['image_id'], batch['samples'], batch['caps_gt']
            with torch.no_grad():
                out = model(samples)
            _, ids = torch.max(out, dim=-1)
            caps_gen = text_field.decode(ids, join_words=False)
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
            image_id, samples, labels, label_masks = batch['image_id'], batch['samples'], batch['labels'], batch['label_masks']

            out = model(samples)
            optim.zero_grad()
            min_len = min(out.shape[1], labels.shape[1])
            labels = labels[:, :min_len].contiguous()
            out = out[:, :min_len].contiguous()
            label_masks = label_masks[:, :min_len].contiguous()

            loss = loss_fn(out, labels, label_masks)
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
            image_id, samples, caps_gt = batch['image_id'], batch['samples'], batch['caps_gt']
            log_probs = model(samples)
            probs, caps = beam_search(log_probs, log_probs.shape[1], beam_size)
            caps_gen = text_field.decode(caps.view(-1, caps.shape[-1]))
            optim.zero_grad()

            # Rewards
            
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(log_probs.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(probs, -1) * (reward - reward_baseline)

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
    multiprocessing.set_start_method('spawn')
    device = torch.device('cuda')
    args = OmegaConf.load('configs/mlnaic.yaml')
    print(args)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.mode, args.exp_name))

    dataloaders, text_field = build_coco_dataloaders(args, device)
    cider_train = Cider()

    if args.dataset.use_cache:
        detector = None
    else:
        detector = build_detector(args)
    encoder = TransformerEncoder(3, text_field.vocab.stoi['<pad>'])
    decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], detector, encoder, decoder).to(device)

    for n, p in model.named_parameters():
        if 'detector' in n:
            p.requires_grad = False

    def lambda_lr(s):
        base_lr = 0.0001
        print("s:", s)
        if s == 0:
            lr = base_lr / 2
        elif s <= 10:
            lr = base_lr
        elif s <= 20:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr

    def lambda_lr_rl(s):
        base_lr = 5e-6
        print("s:", s)
        if s <= 29:
            lr = base_lr
        elif s <= 31:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        return lr

    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)

    loss_fn = MLCrossEntropy()
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    args.model_path = os.path.join("./ckpts", args.mode, args.exp_name)
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
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))
            print('patience:', data['patience'])

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        if not use_rl:
            train_loss = train_xe(model, dataloaders['train'], optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dataloaders['train'], optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
        scheduler.step()
        # Validation loss
        val_loss = evaluate_loss(model, dataloaders['valid'], loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

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
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True
        #####
        if not use_rl:
            if e >= 20:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
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
