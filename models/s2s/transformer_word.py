import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

class Transformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, N_en=3, N_de=3, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(Transformer, self).__init__()
        self.embed_image_word = nn.Sequential( 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(512))
        self.encoder_word = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_en)])
        self.fc_word = nn.Linear(d_model, vocab_size, bias=False)

        self.init_weights()

    def forward(self, images):
        vis, mask = images['grid'], images['mask']
        vis = self.embed_image_word(vis)

        enc_txt = vis
        enc_mask = mask
        for l in self.encoder_word:
            enc_txt = l(enc_txt, enc_txt, enc_txt, enc_mask)
        
        enc_txt_out = enc_txt[:, 0]
        enc_txt_out = self.fc_word(enc_txt_out)
        
        return enc_txt_out

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff