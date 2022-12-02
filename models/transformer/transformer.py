import torch
from torch import nn
from torch.nn import functional as F
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from .attention import ScaledDotProductAttention
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, copy_generator):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        
        self.frame_mask = torch.ones((60,60), dtype=torch.bool).to(device)
        for i in range(60):
            for j in range(-2,3):
                if (i + j)<0 or (i+j)>59:
                    continue
                else:
                    self.frame_mask[i][i+j] = 0.0
                    
        self.fc_geo = torch.nn.Linear(64,1) 
        self.fc_sen = torch.nn.Linear(64,1) 
        self.fc_ocr = torch.nn.Linear(512,1024) 
        
        self.encoder = encoder
        self.decoder = decoder
        self.generator = copy_generator
        
        self.register_state('frame_out', None)
        self.register_state('obj_out', None)
        self.register_state('geo_weights', None)
        self.register_state('sen_weights', None)
        
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, object_feats, geo_weights, sen_weights, seq, ocr_list, ocr_vec, shot_mask, *args):
        batch_size = images.size(0)  
        
        frame_mask = self.frame_mask.repeat((batch_size,1,1)).unsqueeze(1)
        frame_weights = None
        geo_mask = None
        geo_weights = F.relu(self.fc_geo(geo_weights))
        sen_mask = None
        sen_weights = F.relu(self.fc_sen(sen_weights))
        
        frame_out, mask_enc, obj_out, geo_weights, sen_weights = self.encoder(images, object_feats, frame_mask, frame_weights, 
                                                                   geo_mask, geo_weights, sen_mask, sen_weights, shot_mask)
        dec_output, attns, text_att, frame_att, obj_att = self.decoder(seq, frame_out, obj_out, mask_enc,  geo_weights, sen_weights)
        ocr_vec = self.fc_ocr(ocr_vec)
        gener_out, constr_out, context_dec = self.generator(ocr_list, ocr_vec, dec_output, text_att, frame_att, obj_att)
        return gener_out, constr_out, context_dec

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, object_feats, geo_weights, sen_weights, seq, ocr_list, ocr_vec, shot_mask, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                batch_size = visual.size(0)  
                
                frame_mask = self.frame_mask.repeat((batch_size,1,1)).unsqueeze(1)
                frame_weights = None
                geo_mask = None
                geo_weights = F.relu(self.fc_geo(geo_weights))
                sen_mask = None
                sen_weights = F.relu(self.fc_sen(sen_weights))
                
                self.frame_out, self.mask_enc, self.obj_out, self.geo_weights, self.sen_weights = self.encoder(
                    visual, object_feats, frame_mask, frame_weights, geo_mask, geo_weights,sen_mask, sen_weights, shot_mask)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
                
            dec_output, attns, text_att, frame_att, obj_att = self.decoder(it, self.frame_out, self.obj_out, self.mask_enc, self.geo_weights, self.sen_weights)
            ocr_vec = self.fc_ocr(ocr_vec)
            gener_out, _, _ = self.generator(ocr_list, ocr_vec, dec_output, text_att, frame_att, obj_att)
        return gener_out

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
