import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from models.utils import *
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
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

class Transformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, padding_idx, topk, \
                num_timesteps=10, sampling_timesteps=10, ddim_sampling_eta=0.,\
                K_add_word=False, \
                N_en=3, N_wo=6, N_de=3, \
                d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Transformer, self).__init__()
        self.ei_image_emb = nn.Linear(feat_dim, d_model)
        self.ei = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_en)])
        
        self.ew_time_emb = TimestepEmbedder(d_model)
        self.ew = nn.ModuleList([DiffusionDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_wo)])

        self.de_word_emb = nn.Embedding(vocab_size, d_model, padding_idx)
        self.de = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_de)])
        self.de_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.de_fc = nn.Linear(d_model, vocab_size)

        self.vocab_size = vocab_size
        self.seq_len = 20
        self.topk = topk
        self.dim = d_model
        self.label_smoothing = 0.1
        self.confidence = 1.0 - self.label_smoothing
        self.bos_idx = 2
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.K_add_word = K_add_word

        self.betas = sigmoid_beta_schedule(self.num_timesteps) # shape:[num_timesteps,]
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min =1e-20))
        self.timestep_emb = TimestepEmbedder(d_model)
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one
        self.objective = "pred_x0"

        self.init_weights()

        self.ce_loss = nn.NLLLoss(ignore_index=padding_idx)
        self.kl_loss = nn.KLDivLoss(reduction="none")

    def tensor_to(self, device):
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def ce_label_smoothing(self, logit, target):
        if self.label_smoothing>0:
            logit = F.log_softmax(logit, dim=-1)
            logit = logit.view(-1, logit.shape[-1])
            class_num = logit.shape[-1]
            target = target.view(-1)
            target = F.one_hot(target, class_num)
            target = torch.clamp(target.float(), min=self.label_smoothing/(class_num-1), max=1.0-self.label_smoothing)
            loss = -1* torch.sum(target*logit, dim=-1)
        else:
            eps = 1e-12
            loss = -1* logit.gather(1, target.unsqueeze(1)) + torch.log(torch.exp(logit+eps).sum(dim=-1))
        return loss.mean()
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
    
    def forward(self, feat, feat_mask, labels, tokens_kd, ratio=0):
        bs = feat.shape[0]
        device = feat.device
        losses = {}
        feat = self.ei_image_emb(feat)
        for layer in self.ei:
            feat = layer(feat, feat, feat, feat_mask)

        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        gt_topk_emb = self.de_word_emb(gt_topk)

        ew_noise = torch.randn_like(gt_topk_emb)
        ew_batch_times = torch.randint(0, self.num_timesteps, (bs,), device=device)
        ew_x = self.q_sample(x_start=gt_topk_emb, t=ew_batch_times, noise=ew_noise)
        outw = ew_x.to(torch.float32)
        ew_batch_times_emb = self.ew_time_emb(ew_batch_times)
        for layer in self.ew:
            outw = layer(outw, ew_batch_times_emb, feat, feat_mask)
        if self.objective == 'pred_noise':
            ew_target = ew_noise
        elif self.objective == 'pred_x0':
            ew_target = gt_topk_emb
        elif self.objective == 'pred_v':
            ew_v = self.predict_v(gt_topk_emb, ew_batch_times, ew_noise)
            ew_target = ew_v
            
        pos_indx = torch.arange(1, self.seq_len + 1, device=device).view(1, -1)
        out = self.de_pos_emb(pos_indx).repeat(bs, 1, 1)
        for layer in self.de:
            out = layer(out, outw, feat, feat_mask)
            h = self.entropy(out)
            h = h.unsqueeze(-1)
            outw = (1 - h) * out + h * outw
        logit = self.de_fc(out)
        
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1) 
        assign_seq = tokens_kd.view(-1)
        assign_seq[assign_seq < 3] = 0
        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        losses.update({"ew_mse": F.mse_loss(outw, ew_target)})
        losses.update({"de_ce": self.kl_loss(logP, true_dist).sum(1).mean()})
        return losses

    def get_sampling_timesteps(self, batch, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

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
    
    def infer(self, feat, feat_mask):
        bs = feat.shape[0]
        device = feat.device

        feat = self.ei_image_emb(feat)
        for layer in self.ei:
            feat = layer(feat, feat, feat, feat_mask)

        ew_x = torch.randn((bs, self.topk, self.dim), device=device)
        ew_x_start = None
        for t in reversed(range(0, self.num_timesteps)):
            ew_batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
                    
            ew_batched_times_emb = self.ew_time_emb(ew_batched_times)
            for l in self.ew:
                ew_x = l(ew_x, ew_batched_times_emb, feat, feat_mask)

            if self.objective == 'pred_noise':
                ew_pred_noise = ew_x
                ew_x_start = self.predict_start_from_noise(ew_x, ew_batched_times, ew_pred_noise)
            elif self.objective == 'pred_x0':
                ew_x_start = ew_x
                ew_pred_noise = self.predict_noise_from_start(ew_x, ew_batched_times, ew_x_start)
            elif self.objective == 'pred_v':
                ew_v = ew_x
                ew_x_start = self.predict_start_from_v(ew_x, ew_batched_times, ew_v)
                ew_pred_noise = self.predict_noise_from_start(ew_x, ew_batched_times, ew_x_start)
            
            ew_x_start.clamp_(-1., 1.)
            
            ew_model_mean = (
                extract(self.posterior_mean_coef1, ew_batched_times, ew_x.shape) * ew_x_start +
                extract(self.posterior_mean_coef2, ew_batched_times, ew_x.shape) * ew_x
            )
            # posterior_variance = extract(self.posterior_variance, batched_times, x.shape)
            ew_model_log_variance = extract(self.posterior_log_variance_clipped, ew_batched_times, ew_x.shape)
            ew_noise = torch.randn_like(ew_x) if t > 0 else 0. # no noise if t == 0
            ew_x = ew_model_mean + (0.5 * ew_model_log_variance).exp() * ew_noise
        outw = ew_x

        pos_indx = torch.arange(1, self.seq_len + 1, device=device).view(1, -1)
        out = self.de_pos_emb(pos_indx).repeat(bs, 1, 1)
        for layer in self.de:
            out = layer(out, outw, feat, feat_mask)
        logit = self.de_fc(out)
        return F.log_softmax(logit, dim=-1)
    
    def forward_gt(self, feat, feat_mask, labels, tokens_kd, training=True):
        bs = feat.shape[0]
        device = feat.device
        losses = {}
        feat = self.ei_image_emb(feat)
        for layer in self.ei:
            feat = layer(feat, feat, feat, feat_mask)

        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        gt_topk_emb = self.de_word_emb(gt_topk)

        pos_indx = torch.arange(1, self.seq_len + 1, device=device).view(1, -1)
        out = self.de_pos_emb(pos_indx).repeat(bs, 1, 1)
        outd = gt_topk_emb
        for layer in self.de:
            out = layer(out, outd, feat, feat_mask)
        logit = self.de_fc(out)
        
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1)
        if training==False:
            return F.log_softmax(logit, dim=-1)
        assign_seq = tokens_kd.view(-1)
        assign_seq[assign_seq < 3] = 0
        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        losses.update({"de_ce": self.kl_loss(logP, true_dist).sum(1).mean()})
        return losses
    
    def entropy(self, out):
        logit = torch.softmax(self.de_fc(out), -1)
        h = -torch.sum(logit * torch.log(logit), -1) / np.log(logit.shape[-1])
        return h
    
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.dim))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, tensor):
        if len(tensor.shape)==2:
            return torch.matmul(F.one_hot(tensor, num_classes=self.vocab_size).type(torch.float32), self.weight)
        elif len(tensor.shape)==3:
            return torch.matmul(tensor, self.weight.t())
        else:
            raise NotImplementedError
        
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