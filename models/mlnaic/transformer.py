import torch
from torch import nn
from torch.nn import functional as F
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

class Transformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, detector=None, N_en=3, N_de=6, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(Transformer, self).__init__()
        self.detector = detector
        self.embed_image = nn.Sequential( 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(512))
        self.encoder_img = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_en)])
        self.encoder_txt = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_en)])
        self.decoder = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                self_att_module=attention_module, 
                                                enc_att_module=attention_module, 
                                                self_att_module_kwargs=attention_module_kwargs, 
                                                enc_att_module_kwargs=attention_module_kwargs) 
                                                for _ in range(N_de)])    
        
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(100, d_model, 1), freeze=True)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images):
        if self.detector is None:
            gri_feat, gri_mask = images['grid'], images['mask']
            gri_feat = self.embed_image(gri_feat)
        else:
            outputs = self.detector(images)
            gri_feat, gri_mask = outputs['gri_feat'], outputs['gri_mask']
            gri_feat = self.embed_image(gri_feat)
        enc_mask = gri_mask

        enc_img = gri_feat
        for l in self.encoder_img:
            enc_img = l(enc_img, enc_img, enc_img, enc_mask)
        
        enc_txt = gri_feat
        for l in self.encoder_txt:
            enc_txt = l(enc_txt, enc_txt, enc_txt, enc_mask)
        
        enc_fea = self.fc(enc_txt)
        enc_att = torch.mean(enc_fea, 1)

        _, en_ids = torch.max(enc_fea, dim=-1)
        pos_indx = torch.arange(1, en_ids.shape[-1] + 1, device='cuda').view(1, -1)
        out = self.word_emb(en_ids) + self.pos_emb(pos_indx)
        for l in self.decoder:
            out = l(out, out, enc_img, enc_mask)
        out = self.fc(out)

        return F.log_softmax(enc_att, dim=-1), F.log_softmax(out, dim=-1)

    @property
    def d_model(self):
        return self.decoder.d_model

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

    def forward(self, input, input1, enc_output, mask_enc_att, mask=None):
        # MHA+AddNorm
        self_att = self.self_att(input, input1, input1, mask)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        return ff