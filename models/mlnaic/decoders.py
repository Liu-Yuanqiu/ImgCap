import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from models.attention import MultiHeadAttention, sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList
from models.mlnaic.encoders import EncoderLayer

class DecoderLayer(Module):
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

    def forward(self, input, enc_output, mask_enc_att, mask=None):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        return ff


class TransformerDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(100, d_model, 1), freeze=True)
        self.fea2t = nn.Linear(d_model, d_model)

        self.layers = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        # self.layers = ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def get_clip_mat(self, length, size):
        w = torch.zeros((length, size), dtype=torch.float32)
        t = 0.5
        for j in range(length):
            for i in range(size):
                w[j][i] = np.exp(-(j-i*length/size)**2/t)
        w = w / torch.full(w.shape, size, dtype=torch.float32)
        return w
    
    def forward(self, gri_feat, gri_mask, encoder_output, mask_encoder):
        # vocab_weight = self.word_emb.weight

        # en_fea = self.fea2t(encoder_output)
        # en_fea = torch.mean(en_fea, 1)
        # en_att = torch.matmul(en_fea, vocab_weight.t())
        # _, en_ids = torch.topk(en_att, 20, dim=-1)
        en_fea = self.fea2t(encoder_output)
        en_fea = self.fc(en_fea)
        en_att = torch.mean(en_fea, 1)
        # _, en_ids = torch.topk(en_fea, 20, dim=-1)
        _, en_ids = torch.max(en_fea, dim=-1)

        pos_indx = torch.arange(1, en_ids.shape[-1] + 1, device='cuda').view(1, -1)
        out = self.word_emb(en_ids) + self.pos_emb(pos_indx)
        for i, l in enumerate(self.layers):
            logit_now, _ = torch.max(F.log_softmax(self.fc(out), dim=-1), dim=-1)
            mask = None
            if i == 0:
                logit_avg = torch.mean(logit_now, dim=-1, keepdim=True)
                mask = (logit_now < logit_avg).unsqueeze(1).unsqueeze(1)
            elif i == 1:
                logit_avg = torch.mean(logit_now, dim=-1, keepdim=True)
                logit_var = torch.var(logit_now, dim=-1, keepdim=True)
                mask = (logit_now < (logit_avg-logit_var)).unsqueeze(1).unsqueeze(1)
            
            out = l(out, encoder_output, mask_encoder, mask)

            # out = l(out, out, out, mask)
            

        out = self.fc(out)
        return F.log_softmax(en_att, dim=-1), F.log_softmax(out, dim=-1)
