import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from .attention import ScaledDotProductAttention
from .t2t_vit import T2T_module
from .postion_encoder import PositionalEncoding_inside, PositionalEncoding_outside, PositionalEncoding_frame
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        '''
        self.frame_attention = ScaledDotProductAttention(d_model=1536, d_k=64, d_v=64, h=8)
        
        self.tokens_to_token = T2T_module(
                img_size=3, tokens_type='transformer', in_chans=1536, embed_dim=1536)
        '''
        '''
        self.inside_Pe = PositionalEncoding_inside(1536, 0.1, 9)
        self.outside_Pe = PositionalEncoding_outside(1536, 0.1, 30)
        self.frame_Pe = PositionalEncoding_frame(1536, 0.1, 30)
        '''
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
       
        #print(images.shape)
        batch_size = images.size(0)               
        '''
        feats = torch.ones((batch_size,96,1536)).to(device)
        #滑动组帧
        feats_ = torch.zeros((batch_size,30,3,3,1536)).to(device)
        for i in range(batch_size) :
            for j in range(30):
                #（96,1536）->增加组帧内postion（0-8)->reshape（30，3，3，1536）
                
                feats_[i][j]= self.inside_Pe(feats[i][j*3:j*3+9]).reshape((3,3,1536))
        
        feat_T2T = torch.zeros((batch_size,30,1536)).to(device)
        for i in range(30):
            feat_T2T[:,i,:] = self.tokens_to_token(feats_[:,i,:,:,:]).reshape((batch_size,1536))
        #增加组间postion（0-29）再送transformer
        feat_T2T= self.outside_Pe(feat_T2T)
        #增加frame特征
        frame_feats = self.frame_Pe(images)
    
        images = torch.cat((feat_T2T,frame_feats),1)
        print(images.size())        
        '''
        '''
        # 稀疏注意力机制
#         images = self.frame_Pe(images[:,i,:])
        frame_mask = torch.zeros((60,60), dtype=torch.bool).to(device)
        for i in range(60):
            for j in range(-2,3):
                if (i + j)<0 or (i+j)>59:
                    continue
                else:
                    frame_mask[i][i+j] = 1.0
        attention_weights = None
        images = self.frame_attention(images, images, images, frame_mask, attention_weights)
        
        #reshape 做一次T2T
        images = images.reshape((batch_size, 15, 2, 2, 1536))
        
        feat_T2T = torch.zeros((batch_size,15,1536)).to(device)
        for i in range(15):
            feat_T2T[:,i,:] = self.tokens_to_token(images[:,i,:,:,:]).reshape((batch_size,1536))     
        '''
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)
    ''' 
    def get_tokens(self, images):
        batch_size = images.size(0)
        # 稀疏注意力机制
#         images = self.frame_Pe(images[:,i,:])
        frame_mask = torch.zeros((60,60), dtype=torch.bool).to(device)
        for i in range(60):
            for j in range(-2,3):
                if (i + j)<0 or (i+j)>59:
                    continue
                else:
                    frame_mask[i][i+j] = 1.0
        attention_weights = None
        feat_T2T = self.frame_attention(images, images, images, frame_mask, attention_weights)
        #reshape 做一次T2T
        
        images = images.reshape((batch_size, 15, 2, 2, 1536))
        
        feat_T2T = torch.zeros((batch_size,15,1536)).to(device)
        for i in range(15):
            feat_T2T[:,i,:] = self.tokens_to_token(images[:,i,:,:,:]).reshape((batch_size,1536))
        
        return feat_T2T
    ''' 

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
