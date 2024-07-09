import math
import torch
import numpy as np
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.distributions import Categorical
from einops import rearrange, reduce
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

class Transformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, padding_idx, topk, num_timesteps, sampling_timesteps=None, ddim_sampling_eta=0.,\
                K_add_word=False, \
                N_en=3, N_wo=3, N_de=3, \
                d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Transformer, self).__init__()
        self.fea_emb = nn.Sequential( 
            nn.Linear(feat_dim, d_model), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(d_model))
        self.img_emb = nn.Sequential(
            PatchEmbed(),
            # nn.Linear(768, feat_dim)
            )
        # self.pca = PCA(n_component=feat_dim)
        self.encoder_i = nn.ModuleList([DiffusionDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_en)])
        self.encoder_w = nn.ModuleList([DiffusionDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_wo)])    
        self.decoder = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_de)])
        self.word_emb = WordEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.vocab_size = vocab_size
        self.topk = topk
        self.dim = d_model
        self.feat_dim = feat_dim
        self.label_smoothing = 0.1
        self.confidence = 1.0 - self.label_smoothing
        self.bos_idx = 2
        
        self.ce_loss = nn.NLLLoss(ignore_index=padding_idx, reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.init_weights()
        self.num_timesteps = num_timesteps
        self.sampling_timesteps = num_timesteps if sampling_timesteps is None else sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

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
        self.objective = "pred_v" # pred_noise pred_x0
        self.K_add_word = K_add_word
    
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

    def PCA_svd(self, X, k, center=True):
        n, device = X.size()[0], X.device
        ones = torch.ones(n, device=device).view([n,1])
        h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n, device=device).view([n,n])
        H = torch.eye(n, device=device) - h
        # H = H.cuda()s
        X_center =  torch.mm(H.double(), X.double())
        u, s, v = torch.svd(X_center)
        components  = v[:k].t()
        #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
        return components

    def pca(self, X):
        # 计算均值向量并中心化
        mean = torch.mean(X, dim=1, keepdim=True)
        X_centered = X - mean
        
        # 计算协方差矩阵
        cov = torch.matmul(X_centered.transpose(1, 2), X_centered) / (X_centered.size(1) - 1)
        
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = torch.symeig(cov, eigenvectors=True)
        
        # 对特征值和对应的特征向量按降序排列
        # idx = eigenvalues.argsort(descending=True)
        # eigenvalues = eigenvalues[idx]
        # eigenvectors = eigenvectors[:, idx]
        sorted_idx = torch.argsort(eigenvalues, dim=1, descending=True)
        sorted_eigenvalues = torch.gather(eigenvalues, dim=1, index=sorted_idx)
        expanded_sorted_idx = sorted_eigenvalues.unsqueeze(-1).expand_as(eigenvectors).long()
        sorted_eigenvectors = torch.gather(eigenvectors, dim=1, index=expanded_sorted_idx)
        
        # 选择前n_components个特征向量
        principal_components = sorted_eigenvectors[:, :, :self.dim]
        
        # 转换原始数据
        transformed_data = torch.matmul(X_centered, principal_components)
        
        return transformed_data

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
    
    def forward(self, samples, labels, tokens_kd):
        images, gri_feat, grid_mask = samples["image"], samples["grid"], samples["mask"]
        # gri_feat = self.fea_emb(gri_feat)
        images = self.img_emb(images)
        images = self.pca(images)
        gri_feat = self.pca(gri_feat)
        
        bs = gri_feat.shape[0]
        device = gri_feat.device

        losses = {}
        ################# diffusion img ####################
        x_start = self.normalize(gri_feat)
        noise = torch.randn_like(x_start)
        batched_times = torch.randint(0, self.num_timesteps, (bs, ), device=device)
        # batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
        x = self.q_sample(x_start=x_start, t=batched_times, noise=noise)
        outdi = x.to(torch.float32)
        t_emb = self.timestep_emb(batched_times)
        for l in self.encoder_i:
            outdi = l(outdi, t_emb, images)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, batched_times, noise)
            target = v
        loss_i = F.mse_loss(outdi, target)
        losses.update({"img": loss_i})
        outi = self.predict_start_from_v(noise, batched_times, outdi)
        outi = self.unnormalize(outi)

        # outi = self.fea_emb(outi)

        ################# diffusion word ####################
        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        x_start = self.word_emb.id_embed(gt_topk)
        x_start = self.normalize(x_start)
        noise = torch.randn_like(x_start)
        batched_times = torch.randint(0, self.num_timesteps, (bs, ), device=device)
        # batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
        x = self.q_sample(x_start=x_start, t=batched_times, noise=noise)
        outwd = x.to(torch.float32)
        t_emb = self.timestep_emb(batched_times)
        for l in self.encoder_w:
            outwd = l(outwd, t_emb, outi)

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
        outw = self.normalize(outw)

        pos_indx = torch.arange(1, self.topk + 1, device=device).view(1, -1)
        if self.K_add_word:
            out = outw + self.pos_emb(pos_indx).repeat(bs, 1, 1)
            outw = out
        else:
            out = self.pos_emb(pos_indx).repeat(bs, 1, 1)
        for l in self.decoder:
            out = l(out, outw, outi)
        
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
    
    def infer(self, samples):
        images = samples["image"]
        images = self.img_emb(images)
        images = self.pca(images)
        bs, device = images.shape[0], images.device

        x = torch.randn((bs, 60, self.dim), device=device)
        x_start = None
        if self.is_ddim_sampling:
            times = torch.linspace(-1, self.num_timesteps - 1, steps = self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            for time, time_next in time_pairs:
                batched_times = torch.full((bs,), time, device = device, dtype = torch.long)
                batched_times_emb = self.timestep_emb(batched_times)
                model_output = x
                for l in self.encoder_i:
                    model_output = l(model_output, batched_times_emb, images)
                if self.objective == 'pred_noise':
                    pred_noise = model_output
                    x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                elif self.objective == 'pred_x0':
                    x_start = model_output
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
                elif self.objective == 'pred_v':
                    v = model_output
                    x_start = self.predict_start_from_v(x, batched_times, v)
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
                
                if time_next < 0:
                    x = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        else:
            for t in reversed(range(0, self.num_timesteps)):
                batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
                    
                batched_times_emb = self.timestep_emb(batched_times)
                model_output = x
                for l in self.encoder_i:
                    model_output = l(model_output, batched_times_emb, images)

                if self.objective == 'pred_noise':
                    pred_noise = model_output
                    x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                elif self.objective == 'pred_x0':
                    x_start = model_output
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

                elif self.objective == 'pred_v':
                    v = model_output
                    x_start = self.predict_start_from_v(x, batched_times, v)
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
                
                x_start.clamp_(-1., 1.)
                
                model_mean = (
                    extract(self.posterior_mean_coef1, batched_times, x.shape) * x_start +
                    extract(self.posterior_mean_coef2, batched_times, x.shape) * x
                )
                # posterior_variance = extract(self.posterior_variance, batched_times, x.shape)
                model_log_variance = extract(self.posterior_log_variance_clipped, batched_times, x.shape)
                noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
                x = model_mean + (0.5 * model_log_variance).exp() * noise
        outi = self.unnormalize(x)
        # outi = self.fea_emb(outi)
        
        x = torch.randn((bs, self.topk, self.dim), device=device)
        x_start = None
        if self.is_ddim_sampling:
            times = torch.linspace(-1, self.num_timesteps - 1, steps = self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            for time, time_next in time_pairs:
                batched_times = torch.full((bs,), time, device = device, dtype = torch.long)
                batched_times_emb = self.timestep_emb(batched_times)
                model_output = x
                for l in self.encoder_w:
                    model_output = l(model_output, batched_times_emb, outi)
                if self.objective == 'pred_noise':
                    pred_noise = model_output
                    x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                elif self.objective == 'pred_x0':
                    x_start = model_output
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

                elif self.objective == 'pred_v':
                    v = model_output
                    x_start = self.predict_start_from_v(x, batched_times, v)
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
                
                if time_next < 0:
                    x = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
        else:
            for t in reversed(range(0, self.num_timesteps)):
                batched_times = torch.full((bs,), t, device = device, dtype = torch.long)
                    
                batched_times_emb = self.timestep_emb(batched_times)
                model_output = x
                for l in self.encoder_w:
                    model_output = l(model_output, batched_times_emb, outi)

                if self.objective == 'pred_noise':
                    pred_noise = model_output
                    x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                elif self.objective == 'pred_x0':
                    x_start = model_output
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)

                elif self.objective == 'pred_v':
                    v = model_output
                    x_start = self.predict_start_from_v(x, batched_times, v)
                    pred_noise = self.predict_noise_from_start(x, batched_times, x_start)
                
                x_start.clamp_(-1., 1.)
                
                model_mean = (
                    extract(self.posterior_mean_coef1, batched_times, x.shape) * x_start +
                    extract(self.posterior_mean_coef2, batched_times, x.shape) * x
                )
                # posterior_variance = extract(self.posterior_variance, batched_times, x.shape)
                model_log_variance = extract(self.posterior_log_variance_clipped, batched_times, x.shape)
                noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
                x = model_mean + (0.5 * model_log_variance).exp() * noise
        outw = self.unnormalize(x)

        pos_indx = torch.arange(1, self.topk + 1, device=device).view(1, -1)
        if self.K_add_word:
            out = outw + self.pos_emb(pos_indx).repeat(bs, 1, 1)
            outw = out
        else:
            out = self.pos_emb(pos_indx).repeat(bs, 1, 1)
        for l in self.decoder:
            out = l(out, outw, outi)
        
        out = self.word_emb.fc(out)
        return F.log_softmax(out, dim=-1)
    
    def forward_gt(self, images, labels, tokens_kd):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        bs = gri_feat.shape[0]
        device = gri_feat.device

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        losses = {}
        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        outw = self.word_emb.id_embed(gt_topk)

        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
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
    
    def infer_gt(self, images, labels):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
        outw = self.word_emb.id_embed(gt_topk)
        
        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        out = self.word_emb.fc(out)
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

class PatchEmbed(nn.Module):
    """
    Image --> Patch Embedding --> Linear Proj --> Pos Embedding
    Image size -> [224,224,3]
    Patch size -> 16*16
    Patch num -> (224^2)/(16^2)=196
    Patch dim -> 16*16*3 =768
    Patch Embedding: [224,224,3] -> [196,768]
    Linear Proj: [196,768] -> [196,768]
 	Positional Embedding: [197,768] -> [196,768]
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Args:
            img_size: 默认参数224
            patch_size: 默认参数是16
            in_c: 输入的通道数
            embed_dim: 16*16*3 = 768
            norm_layer: 是否使用norm层，默认为否
        """
        super().__init__()
        img_size = (img_size, img_size) # -> img_size = (224,224)
        patch_size = (patch_size, patch_size) # -> patch_size = (16,16)
        self.img_size = img_size # -> (224,224)
        self.patch_size = patch_size # -> (16,16)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # -> grid_size = (14,14)
        self.num_patches = self.grid_size[0] * self.grid_size[1] # -> num_patches = 196
        # Patch+linear proj的这个操作 [224,224,3] --> [14,14,768]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 判断是否有norm_layer层，要是没有不改变输入
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 计算各个维度的大小
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # flatten: [B, C, H, W] -> [B, C, HW], flatten(2)代表的是从2位置开始展开
        # eg: [1,3,224,224] --> [1,768,14,14] -flatten->[1,768,196]
        # transpose: [B, C, HW] -> [B, HW, C]
        # eg: [1,768,196] -transpose-> [1,196,768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
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

import torch
from torch.linalg import eig


class PCA():
    def __init__(self, n_component: int = 1024) -> None:
        """主成分分析

        Args:
            n_component (int): 保留的主成分数
        """
        super().__init__()
        self.n_component = n_component

    def CHECK_SHAPE(self, shape: torch.Size) -> None:
        assert len(shape) >= 2, 'Shape of input is expected bigger than 2!!!'
        limit = 1
        for i in range(1, len(shape)):
            limit *= shape[i]
        assert limit >= self.n_component, f'n_component = {self.n_component}, expected <= {limit}'

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> None:
        """提取主成分

        Args:
            X (torch.Tensor): 待进行主成分分析的输入张量，形状应当为 (batch_size, ...)
        """
        self.CHECK_SHAPE(X.shape)
        Y = X.reshape(X.shape[0], -1).to(X.device)
        self.mean = Y.mean(0)
        Z = Y - self.mean
        
        covariance = Z.T @ Z
        _, eig_vec = eig(covariance)

        self.components = eig_vec[:, :self.n_component]

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """数据降维

        Args:
            X (torch.Tensor): 待降维数据，形状应当为 (batch_size, ...)

        Returns:
            torch.Tensor: 降维后数据
        """
        self.CHECK_SHAPE(X.shape)
        Z = X.reshape(X.shape[0], -1).to(X.device)

        return (Z - self.mean) @ self.components.real
    
    @torch.no_grad()
    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        """高维数据重建

        Args:
            X (torch.Tensor): 待重建数据，形状应当为 (batch_size, ...)

        Returns:
            torch.Tensor: 重建后数据
        """
        assert len(X.shape) == 2, 'Shape of input is expected to equal to 2!!!'

        return (X @ self.components.real.T) + self.mean


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

    def forward(self, input, input1, enc_output, mask_enc_att=None, mask=None, self_att_weight=None):
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

    def forward(self, x, c, vis, vis_mask=None):
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