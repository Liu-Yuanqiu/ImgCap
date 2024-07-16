import clip
import torch
from torch import nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table
from models.s2s.transformer import EncoderLayer, DecoderLayer
class Transformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, se_labels, padding_idx, \
                 K_add_word=False, N_en=3, N_wo=6, N_de=3, \
                 d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Transformer, self).__init__()

        self.feat_emb = nn.Linear(feat_dim, d_model)
        self.ei = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_en)])
        self.ew_word_emb = nn.Embedding(se_labels, d_model, 0)
        self.ew_fc = nn.Linear(d_model, se_labels)
        self.ew = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_de)])
        self.de = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_de)])
        # self.de_word_emb = WordEmbedding(vocab_size, d_model, padding_idx)
        self.de_word_emb = nn.Embedding(vocab_size, d_model, padding_idx)
        self.de_fc = nn.Linear(d_model, vocab_size)
        self.de_pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.topk = 20
        self.label_smoothing = 0.1
        self.confidence = 1.0 - self.label_smoothing
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.asym_loss = AsymmetricLossOptimized(disable_torch_grad_focal_loss=True)
    
    def ce_label_smoothing(self, logit, target, value):
        logP = F.log_softmax(logit.view(-1, logit.shape[-1]), dim=-1) 
        assign_seq = target.view(-1)
        assign_seq[assign_seq < value] = 0
        size = logP.size(1)
        true_dist = logP.clone()
        true_dist.fill_(self.label_smoothing / (size - 1))
        true_dist.scatter_(1, assign_seq.data.unsqueeze(1), self.confidence)
        loss_s = self.kl_loss(logP, true_dist).sum(1).mean()
        return loss_s
    
    def forward(self, feat, feat_mask, labels_gen, labels_gt, tokens_kd):
        bs = feat.shape[0]
        device = feat.device
        losses = {}
        feat = self.feat_emb(feat)
        for l in self.ei:
            feat = l(feat, feat, feat, feat_mask)

        _, w_topk = torch.topk(labels_gen, self.topk, dim=1, largest=True, sorted=True)
        _, w_topk_gt = torch.topk(labels_gt, self.topk, dim=1, largest=True, sorted=True)
        # w_topk_gt[w_topk_gt!=w_topk] = 1
        w_topk_emb = self.ew_word_emb(w_topk)
        outw = w_topk_emb
        for l in self.ew:
            outw = l(outw, outw, feat, feat_mask)
        logit_ew = self.ew_fc(outw)
        
        loss_ew = self.ce_label_smoothing(logit_ew, w_topk_gt, 1)
        losses.update({"ew": loss_ew})
        # outw = torch.cat([w_topk_emb, outw], dim=1)
        logit_ew_ml, _ = torch.max(logit_ew, dim=1)
        loss_ml = self.asym_loss(logit_ew_ml, labels_gt.gt(0).float())
        losses.update({"ml": loss_ml})
        
        pos_indx = torch.arange(1, self.topk + 1, device=device).view(1, -1)
        out = self.de_pos_emb(pos_indx).repeat(bs, 1, 1)
        for l in self.de:
            out = l(out, outw, feat, feat_mask)
        # ce
        logit_d1 = self.de_fc(out)
        
        loss_s = self.ce_label_smoothing(logit_d1, tokens_kd, 3)
        losses.update({"de": loss_s})
        return losses
    
    def infer(self, feat, feat_mask, labels_gen):
        bs = feat.shape[0]
        device = feat.device
        losses = {}
        feat = self.feat_emb(feat)
        for l in self.ei:
            feat = l(feat, feat, feat, feat_mask)

        pos_indx = torch.arange(1, self.topk + 1, device=device).view(1, -1)
        _, w_topk = torch.topk(labels_gen, self.topk, dim=1, largest=True, sorted=True)
        w_topk_emb = self.de_word_emb(w_topk)
        outw = w_topk_emb + self.ew_pos_emb(pos_indx).repeat(bs, 1, 1)
        for l in self.ew:
            outw = l(outw, outw, feat, feat_mask)
        
        out = self.de_pos_emb(pos_indx).repeat(bs, 1, 1)
        for l in self.de:
            out = l(out, outw, feat, feat_mask)
        # ce
        logit = self.de_fc(out)
        return F.log_softmax(logit, dim=-1)

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
    def forward(self, input_ids):
        return torch.matmul(F.one_hot(input_ids, num_classes=self.vocab_size).type(torch.float32), self.weight)
    
    def fc(self, tensor):
        return torch.matmul(tensor, self.weight.t())
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=10, gamma_pos=1, clip=0.2, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        # return -self.loss.sum()
        return - torch.mean(self.loss.sum(-1))