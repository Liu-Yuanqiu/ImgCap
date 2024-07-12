import math
import torch
import numpy as np
from functools import partial
from einops import rearrange, repeat
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.distributions import Categorical
from einops import rearrange, reduce
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table
from .utils import *
class Transformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, padding_idx, topk, num_timesteps, sampling_timesteps=None, ddim_sampling_eta=0.,\
                K_add_word=False, \
                N_en=3, N_wo=6, N_de=3, \
                d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Transformer, self).__init__()
        self.image_emb = nn.Sequential( 
            nn.Linear(feat_dim, d_model), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(d_model))
        self.ei = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_en)])
        self.ew = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_wo)])
        self.ew_bit_emb = BitEmbedding(vocab_size, d_model)
        self.ew_fc = nn.Linear(d_model, vocab_size)
        self.decoder_word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_de)])
        self.decoder_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.decoder_fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.topk = topk
        self.dim = d_model
        self.label_smoothing = 0.1
        self.confidence = 1.0 - self.label_smoothing
        self.bos_idx = 2
        self.K_add_word = K_add_word

        self.ce_loss = nn.NLLLoss(ignore_index=padding_idx, reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.init_weights()
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = num_timesteps if sampling_timesteps is None else sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.time_difference = 0.0

        # self.betas = sigmoid_beta_schedule(self.num_timesteps) # shape:[num_timesteps,]
        # self.alphas = 1. - self.betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.)
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        # self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        # posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min =1e-20))
        # self.timestep_emb = TimestepEmbedder(d_model)
        # self.normalize = normalize_to_neg_one_to_one
        # self.unnormalize = unnormalize_to_zero_to_one
        # self.objective = "pred_v" # pred_noise pred_x0
        
    
    def tensor_to(self, device):
        # self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        # self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        # self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        # self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        # self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        # self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        # self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.ew_bit_emb.vocab_bit_buffer.to(device)
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def q_sample(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def forward(self, images, labels, tokens_kd):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        bs = gri_feat.shape[0]
        device = gri_feat.device

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        losses = {}

        ################# diffusion ####################
        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        x_start = self.word_emb.id_embed(gt_topk)
        x_start = self.normalize(x_start)
        noise = torch.randn_like(x_start)
        batched_times = torch.randint(0, self.num_timesteps, (bs, ), device=device)
        # batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
        x = self.q_sample(x_start=x_start, t=batched_times, noise=noise)
        outwd = x.to(torch.float32)
        t_emb = self.timestep_emb(batched_times)
        for l in self.encoderw:
            outwd = l(outwd, t_emb, enc_img, gri_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, batched_times, noise)
            target = v
        loss_w = F.mse_loss(outwd, target)
        losses.update({"word": loss_w})
        outw = self.predict_start_from_v(noise, batched_times, outwd)
        outw = self.unnormalize(outw)
        # if t>0:
        #     return losses
        ##############################################
        # x_start = None
        # for tt in reversed(range(0, t[0])):
        #     batched_times = torch.full((bs,), tt, device = device, dtype = torch.long)
                
        #     batched_times_emb = self.timestep_emb(batched_times)
        #     model_out = x
        #     for l in self.encoderw:
        #         model_output = l(model_out, batched_times_emb, enc_img, gri_mask)

        #     if self.objective == 'pred_noise':
        #         pred_noise = model_output
        #         x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
        #     elif self.objective == 'pred_x0':
        #         x_start = model_output
        #         pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
        #     elif self.objective == 'pred_v':
        #         v = model_output
        #         x_start = self.predict_start_from_v(x, batched_times, v)
        #         pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

            # loss_w_t = F.mse_loss(model_output, target)
            # loss_w.append(loss_w_t)

            # x_start.clamp_(-1., 1.)
            # model_mean = (
            #     extract(self.posterior_mean_coef1, batched_times, x.shape) * x_start +
            #     extract(self.posterior_mean_coef2, batched_times, x.shape) * x
            # )
            # posterior_variance = extract(self.posterior_variance, batched_times, x.shape)
            # model_log_variance = extract(self.posterior_log_variance_clipped, batched_times, x.shape)
            # noise = torch.randn_like(x) if tt > 0 else 0. # no noise if t == 0
            # x = model_mean + (0.5 * model_log_variance).exp() * noise
        # loss_w = torch.stack(loss_w)
        # losses.update({"word": loss_w.mean()})

        # outw = self.unnormalize(x)
        pos_indx = torch.arange(1, self.topk + 1, device=enc_img.device).view(1, -1)
        if self.K_add_word:
            out = outw + self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
            outw = out
        else:
            out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        # ce
        logit_d1 = self.word_emb.fc(out)
        logP = F.log_softmax(logit_d1.view(-1, logit_d1.shape[-1]), dim=-1) 
        assign_seq = tokens_kd.view(-1)
        assign_seq[assign_seq < 0] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss_s = self.kl_loss(logP, true_dist).sum(1).mean()
        losses.update({"seq": loss_s})
        return losses
    
    def forward_ew(self, images, labels):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        bs = gri_feat.shape[0]
        device = gri_feat.device

        enc_img = gri_feat
        for l in self.ei:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        losses = {}

        ################# diffusion ####################
        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        gt_topk_bit = self.ew_bit_emb.get_bit_repr(gt_topk)
        noise = torch.randn_like(gt_topk_bit)
        times = torch.zeros((bs,), device = device).float().uniform_(0, 0.999)
        noise_level = beta_linear_log_snr(times)
        padded_noise_level = right_pad_dims_to(gt_topk_bit, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        x_start = alpha * gt_topk_bit + sigma * noise

        self_conf = None
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                outwd = self.ew_bit_emb(x_start, times=times)
                for l in self.ew:
                    outwd = l(outwd, outwd, enc_img, gri_mask)
                logit = self.ew_fc(outwd)
                logit_w = self.ew_bit_emb.fc(logit)
                self_conf = logit_w.detach_()

        outwd = self.ew_bit_emb(x_start, times=times, self_conf=self_conf)
        for l in self.ew:
            outwd = l(outwd, outwd, enc_img, gri_mask)
        logit = self.ew_fc(outwd)
        logit_w = self.ew_bit_emb.fc(logit)
        loss_w = F.mse_loss(logit_w, gt_topk_bit)
        losses.update({"mse": loss_w})
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1) 
        assign_seq = gt_topk.view(-1)
        assign_seq[assign_seq < 3] = 0
        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss_s = self.kl_loss(logP, true_dist).sum(1).mean()
        losses.update({"ce": loss_s})
        return losses
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def get_sampling_timesteps(self, batch, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    def infer(self, images):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.ei:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        bs = enc_img.shape[0]
        device = enc_img.device
        time_pairs = self.get_sampling_timesteps(bs, device)
        x = torch.randn((bs, self.topk, self.ew_bit_emb.bit_dim), device=device)
        x_start = None
        for time, time_next in time_pairs:
            time_next = (time_next - self.time_difference).clamp(min=0.)
            noise_cond = beta_linear_log_snr(time)

            outwd = self.ew_bit_emb(x, times=noise_cond, self_conf=x_start)
            for l in self.ew:
                outwd = l(outwd, outwd, enc_img, gri_mask)
            logit = self.decoder_fc(outwd)
            logit_w = self.ew_bit_emb.fc(logit)
            x_start = logit_w
            x_start.clamp_(-self.ew_bit_emb.bit_scale, self.ew_bit_emb.bit_scale)

            log_snr = beta_linear_log_snr(time)
            log_snr_next = beta_linear_log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, x), (log_snr, log_snr_next))
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (x * (1-c) / alpha + c * x_start)
            x = mean
        outw_ids = bits_to_decimal(x, self.vocab_size, self.ew_bit_emb.bit_dim)
        outw_ids = outw_ids.clamp(0., 9489.).long()
        outw = self.decoder_word_emb(outw_ids)
        
        pos_indx = torch.arange(1, self.topk + 1, device=enc_img.device).view(1, -1)
        if self.K_add_word:
            out = outw + self.decoder_pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
            outw = out
        else:
            out = self.decoder_pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder:
            out = l(out, outw, enc_img, gri_mask)
        
        out = self.decoder_fc(out)
        return F.log_softmax(out, dim=-1)
    
    def forward_gt(self, images, labels, tokens_kd):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.ei:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        losses = {}
        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        outw = self.decoder_word_emb(gt_topk)

        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.decoder_pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder:
            out = l(out, outw, enc_img, gri_mask)
        
        # ce
        logit_d1 = self.decoder_fc(out)
        logP = F.log_softmax(logit_d1.view(-1, logit_d1.shape[-1]), dim=-1) 
        assign_seq = tokens_kd.view(-1)
        assign_seq[assign_seq < 3] = 0

        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss_s = self.kl_loss(logP, true_dist).sum(1).mean()
        losses.update({"seq": loss_s})
        return losses
    
    def infer_gt(self, images, labels):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.ei:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        outw = self.decoder_word_emb(gt_topk)
        
        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.decoder_pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder:
            out = l(out, outw, enc_img, gri_mask)
        
        out = self.decoder_fc(out)
        return F.log_softmax(out, dim=-1)
    
    def entropy(self, out):
        logit = torch.softmax(self.word_emb.fc(out), -1)
        h = -torch.sum(logit * torch.log(logit), -1) / np.log(logit.shape[-1])
        return h
    
    def l2_penalty(self, w):
        return torch.sum(w.pow(2)) / 2

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
class BitEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super(BitEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.bit_dim = int(np.ceil(np.log2(vocab_size)))
        self.dim = dim
        self.bit_emb = nn.Linear(self.bit_dim*2, self.dim, bias=False)
        self.bit_scale = 1.
        self.time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim//2),
            nn.Linear(dim//2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        vocab_inds = torch.arange(0, vocab_size).long().view(1, vocab_size, 1, 1)
        self.vocab_bit_buffer = decimal_to_bits(vocab_inds, vocab_size=vocab_size, bits=self.bit_dim) * self.bit_scale
    def get_bit_repr(self, input_ids):
        batch_size, seq_length = input_ids.shape
        input_ids = input_ids.view(batch_size, seq_length, 1, 1) # the same as img: batch_size x channel x height x weight
        input_ids_bit = decimal_to_bits(input_ids, vocab_size=self.vocab_size, bits=self.bit_dim) * self.bit_scale
        return input_ids_bit
    
    def fc(self, logit):
        bs = logit.shape[0]
        buffer_probs = nn.Softmax(-1)(logit)
        self.vocab_bit_buffer = self.vocab_bit_buffer.to(logit.device)
        logit_w = torch.matmul(buffer_probs, self.vocab_bit_buffer.expand(bs, -1, -1))
        return logit_w
    def forward(self, input_ids_bit, self_conf=None, times=None):
        if len(input_ids_bit.shape) == 2:
            input_ids_bit = self.get_bit_repr(input_ids_bit)
        if self_conf is None:
            self_conf = torch.zeros_like(input_ids_bit)
        input_ids_bit = torch.cat([self_conf, input_ids_bit], dim=-1)
        input_ids_bit_emb = self.bit_emb(input_ids_bit)
        if times is not None:
            input_ids_bit_emb = input_ids_bit_emb + self.time_emb(times).unsqueeze(1)
        return input_ids_bit_emb

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.dim))
        # self.bit_dim = int(np.ceil(np.log2(vocab_size)))
        # self.weight_bit = nn.Parameter(torch.randn(self.bit_dim, self.dim))
        # vocab_inds = torch.arange(0, vocab_size).long().view(1, vocab_size, 1, 1)
        # self.vocab_bit_buffer = self.decimal_to_bits(vocab_inds, vocab_size=vocab_size, bits=self.bit_dim).cuda()
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def id_embed(self, input_ids):
        return torch.matmul(F.one_hot(input_ids, num_classes=self.vocab_size).type(torch.float32), self.weight)
    
    # def bit_embed(self, input_ids):
    #     input_ids_bit = self.get_bit_repr(input_ids)
    #     return torch.matmul(input_ids_bit, self.weight_bit)
    
    def fc(self, tensor):
        return torch.matmul(tensor, self.weight.t())
    
    # def bit_fc(self, tensor):
    #     logit = torch.matmul(tensor, self.weight.t())
    #     prob = F.softmax(logit, -1)
    #     # prob = F.sigmoid(logit)
    #     logit_w = torch.matmul(prob, self.vocab_bit_buffer.expand(prob.shape[0], -1, -1))
    #     return logit, logit_w
    
    def toText(self, tensor):
        prob = F.softmax(torch.matmul(tensor, self.weight.t()), -1)
        return torch.matmul(prob, self.weight)

    # def get_bit_repr(self, input_ids):
    #     batch_size, seq_length = input_ids.shape
    #     input_ids = input_ids.view(batch_size, seq_length, 1, 1) # the same as img: batch_size x channel x height x weight
    #     input_ids_bit = self.decimal_to_bits(input_ids, vocab_size=self.vocab_size, bits=self.bit_dim)
    #     return input_ids_bit
    
    # def decimal_to_bits(self, x, vocab_size, bits):
    #     """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    #     device = x.device

    #     x = x.clamp(0, vocab_size-1)

    #     mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
    #     mask = rearrange(mask, 'd -> d 1 1')
    #     x = rearrange(x, 'b c h w -> b c 1 h w')

    #     bits = ((x & mask) != 0).float()
    #     # bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    #     bits = bits.squeeze(-1).squeeze(-1) # batch_size x seq_length x bits x 1 x 1 -> batch_size x seq_length x bits
    #     bits = bits * 2 - 1
    #     return bits

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm1(queries + self.dropout1(att))
        ff = self.pwff(att)
        out = self.lnorm2(att + self.dropout2(ff))
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)

    def forward(self, input, input1, enc_output, mask_enc_att, mask=None, self_att_weight=None):
        # MHA+AddNorm
        self_att = self.self_att(input, input1, input1, mask, self_att_weight)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        out = self.lnorm3(enc_att + self.dropout3(ff))
        return out
    
class DiffusionDecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DiffusionDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )

    def forward(self, x, c, vis, vis_mask):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_pf, scale_pf, gate_pf = self.adaLN_modulation(c).chunk(9, dim=1)
        input = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_att(input, input, input)
        input = modulate(self.norm2(x), shift_mca, scale_mca)
        x = x + gate_mca.unsqueeze(1) * self.enc_att(input, vis, vis, vis_mask)
        input = modulate(self.norm3(x), shift_pf, scale_pf)
        x = x + gate_pf.unsqueeze(1) * self.pwff(input)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float32) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def normalize_to_neg_one_to_one(t):
    return t * 2 - 1

if __name__ == '__main__':
    print(sigmoid_beta_schedule(100))