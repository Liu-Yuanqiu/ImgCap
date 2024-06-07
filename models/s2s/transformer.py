import torch
import numpy as np
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.distributions import Categorical
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

class Transformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, padding_idx, topk, \
                teacher_model=None, use_loss_word=False, use_loss_ce=True, \
                use_loss_l2=False, use_loss_entropy=False, use_loss_kl=False, \
                N_en=3, N_wo=3, N_de=3, \
                d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, \
                identity_map_reordering=False, attention_module=None, \
                attention_module_kwargs=None):
        super(Transformer, self).__init__()
        self.image_emb = nn.Sequential( 
            nn.Linear(feat_dim, d_model), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(d_model))
        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_en)])
        self.img2word = nn.Linear(d_model, d_model)
        self.encoderw = ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_wo)])    
        self.decoder1 = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                self_att_module=attention_module, 
                                                enc_att_module=attention_module, 
                                                self_att_module_kwargs=attention_module_kwargs, 
                                                enc_att_module_kwargs=attention_module_kwargs) 
                                                for _ in range(N_de)])    
        self.decoder2 = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                self_att_module=attention_module, 
                                                enc_att_module=attention_module, 
                                                self_att_module_kwargs=attention_module_kwargs, 
                                                enc_att_module_kwargs=attention_module_kwargs) 
                                                for _ in range(N_de)])
        self.word_emb = WordEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.vocab_size = vocab_size
        self.topk = topk
        self.teacher_model = teacher_model
        self.use_loss_word = use_loss_word
        self.use_loss_ce = use_loss_ce
        self.use_loss_l2 = use_loss_l2
        self.use_loss_entropy = use_loss_entropy
        self.use_loss_kl = use_loss_kl
        self.bos_idx = 2
        
        self.ce_loss = nn.NLLLoss(ignore_index=padding_idx, reduction='none')
        self.kl_loss = nn.KLDivLoss()

        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, labels, tokens_kd):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        if self.teacher_model is None:
            enc_txt = self.img2word(enc_img)
            for l in self.encoderw:
                enc_txt = l(enc_txt, enc_txt, enc_img, gri_mask)
            enc_txt = enc_txt[:, :self.topk]
            outw = self.word_emb.toText(enc_txt)
        else:
            _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            outw = self.word_emb.embed(gt_topk)

        # mse_loss = self.mse_loss(out1, label_emb)
        # enc_txt_out = enc_img[:, 0]
        # enc_txt_out = self.img2word(enc_txt_out)
        # word_logit = self.fc_word(enc_txt_out)

        # with torch.no_grad():
        #     offline_logit = torch.nn.functional.sigmoid(word_logit.detach())
        #     prob, pred_topk = offline_logit.topk(self.topk, dim=1, largest=True, sorted=True)
        # if labels is not None and gen_tag_ratio is not None:
        #     batch_len = int((1 - gen_tag_ratio) * self.topk)
        #     _, gt_topk = torch.topk(labels, batch_len, dim=1, largest=True, sorted=True)
        #     pred_topk = pred_topk[:, batch_len:]
        #     pred_topk = torch.cat([gt_topk, pred_topk], dim=1)
            # pred_topk = pred_topk[:, :self.topk]
        
        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        self_att_weight = torch.log(torch.tensor(2.)) - self.entropy(out)
        self_att_weight = self_att_weight.unsqueeze(1).unsqueeze(1)
        for l in self.decoder2:
            out = l(out, out, enc_img, gri_mask, self_att_weight=self_att_weight)
        outw = self.word_emb.fc(outw)
        outs = self.word_emb.fc(out)
        
        # loss
        loss = {}
        # word predict
        if self.use_loss_word:
            prob, target = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            mask = (prob > 0).float().view(-1)
            logit_w = F.log_softmax(outw, dim=-1)
            loss_w = self.ce_loss(logit_w.view(-1, self.vocab_size), target.view(-1))
            loss_w = torch.sum(loss_w*mask, -1) / torch.sum(mask, -1)
            loss['word'] = loss_w
        # ce
        if self.use_loss_ce:
            logit = F.log_softmax(outs, dim=-1)
            loss_ce = self.ce_loss(logit.view(-1, self.vocab_size), tokens_kd.view(-1))
            loss_ce = loss_ce.mean()
            loss['ce'] = loss_ce
        # l2
        if self.use_loss_l2:
            loss_l2 = self.l2_penalty(self.word_emb.weight)
            loss['l2'] = loss_l2
        # entropy
        if self.use_loss_entropy:
            loss_e = self.entropy(out)
            loss['entropy'] = loss_e.mean()
        # kl
        if self.teacher_model is not None and self.use_loss_kl:
            logit_tm = self.teacher_model.infer(images)
            logit_tm = torch.exp(logit_tm)
            loss_kl = self.kl_loss(logit, logit_tm)
            loss['loss_kl'] = loss_kl
        return loss
    
    def infer(self, images, labels=None):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        if self.teacher_model is None:
            enc_txt = self.img2word(enc_img)
            for l in self.encoderw:
                enc_txt = l(enc_txt, enc_txt, enc_img, gri_mask)
            enc_txt = enc_txt[:, :self.topk]
            outw = self.word_emb.toText(enc_txt)
        else:
            _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            outw = self.word_emb.embed(gt_topk)
        
        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        self_att_weight = torch.log(torch.tensor(2.)) - self.entropy(out)
        self_att_weight = self_att_weight.unsqueeze(1).unsqueeze(1)
        for l in self.decoder2:
            out = l(out, out, enc_img, gri_mask, self_att_weight=self_att_weight)
        outw = self.word_emb.fc(outw)
        out = self.word_emb.fc(out)
        return F.log_softmax(outw, dim=-1), F.log_softmax(out, dim=-1)
    
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

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def embed(self, tenosr):
        return torch.matmul(F.one_hot(tenosr, num_classes=self.vocab_size).type(torch.float32), self.weight)
    
    def fc(self, tensor):
        return torch.matmul(tensor, self.weight.t())
    
    def toText(self, tensor):
        prob = F.softmax(torch.matmul(tensor, self.weight.t()), -1)
        return self.norm(torch.matmul(prob, self.weight))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm1(queries + self.dropout1(att))
        ff = self.pwff(att)
        out = self.lnorm2(att + self.dropout2(ff))
        return out
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

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