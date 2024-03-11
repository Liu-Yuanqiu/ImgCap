import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from models.attention import MultiHeadAttention, sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


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

    def forward(self, input, enc_output, mask_enc_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input)
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
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        # self.pos_emb = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)

        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module, enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs, enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, encoder_output, mask_encoder):
        vocab_weight = self.word_emb.weight
        vocab_prob = encoder_output @ vocab_weight.t()
        vocab_prob = vocab_prob.masked_fill(mask_encoder.squeeze().unsqueeze(-1), -np.inf)
        input = torch.sigmoid(vocab_prob) @ vocab_weight


        out = self.lnorm(encoder_output + self.dropout(input))
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
