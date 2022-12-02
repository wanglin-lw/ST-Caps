import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F, Embedding, DataParallel
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList

from numpy import inf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CopyGenerator(Module):

    def __init__(self, vocab_size, d_model):
        super(CopyGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.gen_proj = Linear(d_model, vocab_size)
        self.prob_proj = Linear(d_model, 1)
                
        self.fc_decoder = nn.Linear(d_model, d_model)
        self.fc_text = nn.Linear(d_model, d_model)
        self.fc_frame = nn.Linear(d_model, d_model)
        self.fc_obj = nn.Linear(d_model, d_model)
        self.fc_context = nn.Linear(d_model, d_model)
        
    def forward(self, src_ocr, ocr_vec, decode_output, text_att, frame_att, obj_att):
        """
        Generate final vocab distribution.
        :param src_ocr: [B, ocr_len]
        :param ocr_vec: [B, ocr_len, d_model]
        :param decode_output: [B, T, H]
        :return:
        """
        batch_size = decode_output.size(0)
        steps = decode_output.size(1)
        seq = src_ocr.size(1)
        
        src_ocr = src_ocr.type(torch.long)
        src_ocr = src_ocr.unsqueeze(1).repeat([1, steps, 1]) 

        
        decode_attn = torch.matmul(decode_output, ocr_vec.transpose(1, 2)) 
        context = torch.matmul(decode_attn, ocr_vec) 
        prob = torch.sigmoid(self.prob_proj(context)) 
        
        context_dec = torch.matmul(decode_output, context.transpose(1, 2))/1000
        context_dec = torch.sigmoid(context_dec) 
        context_dec = torch.log(context_dec)

        gen_logits = prob * torch.softmax(self.gen_proj(decode_output), dim=-1)

        copy_logits = torch.zeros_like(gen_logits)
        copy_logits = copy_logits.scatter_add(2, src_ocr, decode_attn) 
        copy_logits = torch.softmax(copy_logits, dim=-1) 

        copy_logits = (1 - prob) * copy_logits
        
        decode_output = self.fc_decoder(decode_output)
        text_att = self.fc_text(text_att)
        frame_att = self.fc_frame(frame_att)
        obj_att = self.fc_obj(obj_att)
        context = self.fc_context(context)
        decode_output = torch.nn.functional.normalize(decode_output,3) 
        context = torch.nn.functional.normalize(context,3) 
        text_att = torch.nn.functional.normalize(text_att,3)
        frame_att = torch.nn.functional.normalize(frame_att,3)
        obj_att = torch.nn.functional.normalize(obj_att,3)

        decode_output_sqrt = torch.sqrt(torch.sum(decode_output*decode_output, axis=2))
        text_att_sqrt = torch.sqrt(torch.sum(text_att*text_att, axis=2))*decode_output_sqrt
        frame_att_sqrt = torch.sqrt(torch.sum(frame_att*frame_att, axis=2))*decode_output_sqrt
        obj_att_sqrt = torch.sqrt(torch.sum(obj_att*obj_att, axis=2))*decode_output_sqrt
        context_ocr_sqrt = torch.sqrt(torch.sum(context*context, axis=2))*decode_output_sqrt
        
        constr_out = torch.sum(decode_output.unsqueeze(2).repeat([1, 1, 4, 1])*
                            torch.stack([text_att, frame_att, obj_att, context],axis=2),dim=3)  
        sqrt_res = torch.stack([text_att_sqrt, frame_att_sqrt, obj_att_sqrt, context_ocr_sqrt],axis=2)
        constr_out= constr_out / sqrt_res
    
        constr_out = torch.log(torch.softmax(constr_out*0.01, dim=2)) 

        return torch.log(copy_logits + gen_logits), constr_out, context_dec
