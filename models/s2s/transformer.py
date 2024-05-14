import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.containers import ModuleList
from models.attention import MultiHeadAttention, PositionWiseFeedForward, sinusoid_encoding_table

class Transformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, topk, N_en=3, N_de=3, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(Transformer, self).__init__()
        self.image_emb = nn.Sequential( 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(p=0.1), 
            nn.LayerNorm(512))
        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                       identity_map_reordering=identity_map_reordering, 
                                                       attention_module=attention_module, 
                                                       attention_module_kwargs=attention_module_kwargs) 
                                                       for _ in range(N_en)])
        self.img2word = nn.Linear(d_model, d_model)
        self.decoder = ModuleList([DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, 
                                                self_att_module=attention_module, 
                                                enc_att_module=attention_module, 
                                                self_att_module_kwargs=attention_module_kwargs, 
                                                enc_att_module_kwargs=attention_module_kwargs) 
                                                for _ in range(N_de)])    
        
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(200, d_model, 1), freeze=False)
        self.fc_word = nn.Linear(d_model, vocab_size, bias=False)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.topk = topk

    def forward_word(self, images):
        vis, mask = images['grid'], images['mask']
        vis = self.image_emb(vis)
        enc_txt = vis
        enc_mask = mask
        for l in self.encoder:
            enc_txt = l(enc_txt, enc_txt, enc_txt, enc_mask)
        
        enc_txt_out = enc_txt[:, 0]
        enc_txt_out = self.img2word(enc_txt_out)
        enc_txt_out = self.fc_word(enc_txt_out)
        return enc_txt_out
    
    def forward(self, images, labels=None, gen_tag_ratio=None):
        gri_feat, gri_mask = images['grid'], images['mask']
        gri_feat = self.image_emb(gri_feat)

        enc_img = gri_feat
        for l in self.encoder:
            enc_img = l(enc_img, enc_img, enc_img, gri_mask)

        enc_txt_out = enc_img[:, 0]
        enc_txt_out = self.img2word(enc_txt_out)
        word_logit = self.fc_word(enc_txt_out)

        with torch.no_grad():
            offline_logit = torch.nn.functional.sigmoid(word_logit.detach())
            prob, pred_topk = offline_logit.topk(self.topk, dim=1, largest=True)
        if labels is not None and gen_tag_ratio is not None:
            # fuse the generated tags with GT tags at specific portion X%
            for batch_idx, lab in enumerate(labels):
                batch_tag = torch.nonzero(lab, as_tuple=False).squeeze(1)
                batch_len = min(int((1 - gen_tag_ratio) * len(batch_tag)), self.topk)
                indices = torch.randperm(batch_len)
                batch_tag = batch_tag[indices]
                pred_topk[batch_idx, :batch_len] = batch_tag

        out1 = self.word_emb(pred_topk)
        pos_indx = torch.arange(1, enc_img.shape[1] + 1, device='cuda').view(1, -1)
        out = self.pos_emb(pos_indx).repeat(enc_img.shape[0], 1, 1)
        for l in self.decoder:
            out = l(out, out1, enc_img, gri_mask)
        out = self.fc(out)
        
        return F.log_softmax(out, dim=-1)

    def entropy(self, out):
        logit = torch.softmax(self.fc(out), -1)
        h = -torch.sum(logit * torch.log(logit), -1)
        return h
    
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