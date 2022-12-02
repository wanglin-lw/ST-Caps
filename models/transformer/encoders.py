from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderLayer(nn.Module):
    def __init__(self, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.obj_mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=attention_module,
                                          attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, obj_q, obj_k, obj_v, 
                attention_mask=None, attention_weights=None, 
                geo_mask=None, geo_weights=None, sen_mask=None, sen_weights=None, shot_mask=None):

        ff, _ = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        geo_att, geo_weights = self.obj_mhatt(obj_q, obj_k, obj_v, geo_mask, geo_weights)
        
        sen_att, sen_weights = self.obj_mhatt(obj_q, obj_k, obj_v, sen_mask, sen_weights)

        alpha1 = torch.sigmoid(geo_att)
        alpha2 = torch.sigmoid(sen_att)
        obj_ff = (geo_att * alpha1 + sen_att * alpha2) / np.sqrt(2)
        
        shot_mask = shot_mask.bool()
        enc_att1, _ = self.cross_att(ff, obj_ff, obj_ff, shot_mask.permute(0, 2, 1).unsqueeze(1))
        enc_att2, _ = self.cross_att(obj_ff, ff, ff, shot_mask.unsqueeze(1))
        enc_att1 = self.pwff(enc_att1)
        enc_att2 = self.pwff(enc_att2)

        return enc_att1, enc_att2, geo_weights, sen_weights 


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, object_feats, attention_mask=None, attention_weights=None,  geo_mask=None, geo_weights=None, sen_mask=None, sen_weights=None, shot_mask=None):
        # input (b_s, seq_len, d_in)
        if attention_mask==None:
            # mask the padding frame if there has padding frame or mask is all of false
            attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1) 
        outs = []
        out = input
        
        if geo_mask == None:
            geo_mask = (torch.sum(object_feats, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1) 
        if sen_mask == None:
            sen_mask = (torch.sum(object_feats, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1) 
        obj_outs = []
        obj_out = object_feats
        
        
        for l in self.layers:
            out, obj_out, geo_weights, sen_weights  = l(out, out, out, obj_out, obj_out, obj_out,
                             attention_mask, attention_weights, geo_mask, geo_weights, sen_mask, sen_weights, shot_mask)
            
            outs.append(out.unsqueeze(1))
            obj_outs.append(obj_out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        obj_outs = torch.cat(obj_outs, 1)
        return outs, attention_mask, obj_outs, geo_weights, sen_weights


class Encoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(Encoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, object_feats, attention_mask=None, attention_weights=None, geo_mask=None, geo_weights=None, sen_mask=None, sen_weights=None, shot_mask=None):

        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        obj_out = F.relu(self.fc(object_feats))
        obj_out = self.dropout(obj_out)
        obj_out = self.layer_norm(obj_out)

        return super(Encoder, self).forward(out, obj_out, 
                        attention_mask=attention_mask, attention_weights=attention_weights,
                        geo_mask=geo_mask, geo_weights=geo_weights, sen_mask=sen_mask, sen_weights=sen_weights, 
                        shot_mask = shot_mask,)

