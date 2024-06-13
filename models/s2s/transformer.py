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
    def __init__(self, feat_dim, vocab_size, padding_idx, topk, \
                use_loss_word=False, use_loss_ce=True, \
                use_loss_entropy=False, \
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
        self.pos_emb_freeze = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=True)
        self.vocab_size = vocab_size
        self.topk = topk
        self.use_loss_word = use_loss_word
        self.use_loss_ce = use_loss_ce
        self.label_smoothing = 0.1
        self.confidence = 1.0 - self.label_smoothing
        self.use_loss_entropy = use_loss_entropy
        self.bos_idx = 2
        
        self.ce_loss = nn.NLLLoss(ignore_index=padding_idx, reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="none")
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

        losses = {}
        if self.use_loss_word:
            enc_txt = self.img2word(enc_img)
            enc_txt = self.word_emb.toText(enc_txt)
            for l in self.encoderw:
                enc_txt = l(enc_txt, enc_txt, enc_img, gri_mask)
            outw = enc_txt[:, :self.topk]

            _, logit_w = self.word_emb.bit_fc(outw)
            _, targets = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            targets = self.word_emb.get_bit_repr(targets)
            loss_w = F.mse_loss(logit_w, targets)
            losses.update({"word": loss_w})
        else:
            _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            outw = self.word_emb.id_embed(gt_topk)

        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        # ce
        if self.use_loss_ce:
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

        # out = out + self.pos_emb_freeze(pos_indx)
        # self_att_weight = torch.log(torch.tensor(2.)) - self.entropy(out)
        # self_att_weight = self_att_weight.unsqueeze(1).unsqueeze(1)
        # for l in self.decoder2:
        #     out = l(out, out, enc_img, gri_mask, self_att_weight=self_att_weight)
        # # ce
        # if self.use_loss_ce:
        #     out_d2 = self.word_emb.fc(out)
        #     logit_d2 = F.log_softmax(out_d2, dim=-1)
        #     loss_ce = self.ce_loss(logit_d2.view(-1, self.vocab_size), tokens_kd.view(-1))
        #     loss_ce = loss_ce.mean()
        #     loss['ce2'] = loss_ce
        

        # entropy
        if self.use_loss_entropy:
            loss_e = self.entropy(out)
            losses.update({"entropy": loss_e.mean()})
        return losses   

    def infer(self, images, labels=None):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        if labels is None:
            enc_txt = self.img2word(enc_img)
            enc_txt = self.word_emb.toText(enc_txt)
            for l in self.encoderw:
                enc_txt = l(enc_txt, enc_txt, enc_img, gri_mask)
            outw = enc_txt[:, :self.topk]
        else:
            _, gt_topk = torch.topk(labels, self.topk, dim=1, largest=True, sorted=True)
            outw = self.word_emb.id_embed(gt_topk)
        
        pos_indx = torch.arange(1, self.topk + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder1:
            out = l(out, outw, enc_img, gri_mask)
        
        # self_att_weight = torch.log(torch.tensor(2.)) - self.entropy(out)
        # self_att_weight = self_att_weight.unsqueeze(1).unsqueeze(1)
        # for l in self.decoder2:
        #     out = l(out, out, enc_img, gri_mask, self_att_weight=self_att_weight)
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

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.dim))
        self.bit_dim = int(np.ceil(np.log2(vocab_size)))
        self.weight_bit = nn.Parameter(torch.randn(self.bit_dim, self.dim))
        vocab_inds = torch.arange(0, vocab_size).long().view(1, vocab_size, 1, 1)
        self.vocab_bit_buffer = self.decimal_to_bits(vocab_inds, vocab_size=vocab_size, bits=self.bit_dim).cuda()
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def id_embed(self, input_ids):
        return torch.matmul(F.one_hot(input_ids, num_classes=self.vocab_size).type(torch.float32), self.weight)
    
    def bit_embed(self, input_ids):
        input_ids_bit = self.get_bit_repr(input_ids)
        return torch.matmul(input_ids_bit, self.weight_bit)
    
    def fc(self, tensor):
        return torch.matmul(tensor, self.weight.t())
    
    def bit_fc(self, tensor):
        logit = torch.matmul(tensor, self.weight.t())
        prob = F.softmax(logit, -1)
        # prob = F.sigmoid(logit)
        logit_w = torch.matmul(prob, self.vocab_bit_buffer.expand(prob.shape[0], -1, -1))
        return logit, logit_w
    
    def toText(self, tensor):
        prob = F.softmax(torch.matmul(tensor, self.weight.t()), -1)
        return torch.matmul(prob, self.weight)

    def get_bit_repr(self, input_ids):
        batch_size, seq_length = input_ids.shape
        input_ids = input_ids.view(batch_size, seq_length, 1, 1) # the same as img: batch_size x channel x height x weight
        input_ids_bit = self.decimal_to_bits(input_ids, vocab_size=self.vocab_size, bits=self.bit_dim)
        return input_ids_bit
    
    def decimal_to_bits(self, x, vocab_size, bits):
        """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
        device = x.device

        x = x.clamp(0, vocab_size-1)

        mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b c h w -> b c 1 h w')

        bits = ((x & mask) != 0).float()
        # bits = rearrange(bits, 'b c d h w -> b (c d) h w')
        bits = bits.squeeze(-1).squeeze(-1) # batch_size x seq_length x bits x 1 x 1 -> batch_size x seq_length x bits
        bits = bits * 2 - 1
        return bits


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