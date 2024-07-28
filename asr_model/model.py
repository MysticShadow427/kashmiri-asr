import torch
from torch import nn
import numpy as np
import logging
from conformer import Conformer
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from transformers import AutoModel

class ASR_Model(nn.Module):
    def __init__(self,
        mel_dim,
        cdepth ,
        cdim_head ,
        cheads ,
        cff_mult, 
        conv_expansion_factor, 
        conv_kernel_size ,
        cattn_dropout,
        cff_dropout ,
        conv_dropout ,
        hidden_size = 192, # lstm
        d_model = 768, # transformer input size
        nhead = 4, # transformer
        dim_ff = 768,
        num_layers = 1,
        tembedding_dim = None, # mostly 768
        num_vocab = None,
        text_decoder = None,
        xlstm_cfg  = None,
        dropout = 0.1,
        bidirectional = False,
        tencoder = 'new'
        ):
        super().__init__()
        self.encoder = Conformer(
            mel_dim,
            cdepth ,
            cdim_head ,
            cheads ,
            cff_mult, 
            conv_expansion_factor, 
            conv_kernel_size ,
            cattn_dropout,
            cff_dropout ,
            conv_dropout
        )

        self.lm_head = nn.Linear(tembedding_dim,num_vocab)
        self.bidirectional = bidirectional
        self.text_encoder = None
        self.tencoder = tencoder
        if tencoder != 'pretrained':
            self.text_encoder = AutoModel.from_pretrained('ai4bharat/IndicBERTv2-SS',return_dict=False)
        else:
            self.text_encoder = AutoModel.from_pretrained('text_encoder.pth',return_dict = False)
        
        modules = [self.text_decoder.embeddings]
        for module in modules:
            for param in module.parameters():
                param.required_grad = False
        
        self.to_q = None
        self.to_k = None
        self.to_v = None 
        self.cross_attn = None
        self.text_decoder = text_decoder
        self.lm = None

        self.project = None
        if bidirectional:
            self.project = nn.Linear(in_features = hidden_size,out_features=768) 

        if text_decoder == 'xlstm':
            self.lm = xLSTMBlockStack(xlstm_cfg)
            self.to_q = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.to_k = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.to_v = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.cross_attn = nn.MultiheadAttention(embed_dim=768,num_heads=nhead,dropout=dropout,batch_first=True)
        elif text_decoder == 'bilstm' or 'lstm':
            self.lm = nn.LSTM(tembedding_dim,hidden_size,num_layers,True,dropout,True,bidirectional)
            self.to_q = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.to_k = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.to_v = nn.Linear(tembedding_dim,tembedding_dim//4)
            self.cross_attn = nn.MultiheadAttention(embed_dim=768,num_heads=nhead,dropout=dropout,batch_first=True)
        else : # transformer
            self.decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_ff,dropout,'relu',1e-5,True,False,True)
            self.lm = nn.TransformerDecoder(self.decoder_layer,num_layers) 
           
    
    def generate_square_subsequent_mask(sz):
        """for tgt_mask in transformer decoder"""
        mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self,mel_input_ids,tgt_ids,padding_mask):
        mem = self.encoder(mel_input_ids)
        tgt = self.text_encoder(tgt_ids,padding_mask)
        batch_size = mel_input_ids.shape[0]
    
        if self.text_decoder == 'bilstm':
            tgt = self.project(tgt)
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
            q = self.to_q(tgt)
            k = self.to_k(mem)
            v = self.to_v(mem)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            attn_out = self.cross_attn(q,k,v,key_padding_mask = padding_mask,weights = False,attn_mask=tgt_mask,is_causal = True)
            outputs, (hidden, cell) = self.lm(attn_out, (h0, c0))
            logits = self.lm_head(outputs)
            return logits
        elif self.text_decoder == 'lstm':
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to('cuda')
            q = self.to_q(tgt)
            k = self.to_k(mem)
            v = self.to_v(mem)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            attn_out = self.cross_attn(q,k,v,key_padding_mask = padding_mask,weights = False,attn_mask=tgt_mask,is_causal = True)
            outputs, (hidden, cell) = self.lm(attn_out, (h0, c0))
            logits = self.lm_head(outputs)
            return logits
        elif self.text_decoder == 'xlstm':
            q = self.to_q(tgt)
            k = self.to_k(mem)
            v = self.to_v(mem)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            attn_out = self.cross_attn(q,k,v,key_padding_mask = padding_mask,weights = False,attn_mask=tgt_mask,is_causal = True)
            outputs = self.lm(attn_out)
            logits = self.lm_head(outputs)
            return logits
        else:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            outputs = self.lm(mem,tgt,tgt_mask=tgt_mask,tgt_key_padding_mask=padding_mask,tgt_is_causal=True)
            logits = self.lm_head(outputs)
            return logits # need to be [batch_sz,num_tim_steps,num_vocab] 








