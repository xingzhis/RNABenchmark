from structure.resnet import renet_b16
from transformers.models.esm.modeling_esm import *
import torch
import torch.nn as nn
import pdb
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)


class SSCNNPredictor(nn.Module):
    def __init__(self, args, extractor, config, tokenizer, is_freeze = False):
        super(SSCNNPredictor, self).__init__()
        self.extractor = extractor
        self.config = config
        self.tokenizer = tokenizer
        self.cnn = renet_b16(myChannels=config.hidden_size, bbn=16)
        self.args = args

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()
        if self.args.token_type == 'bpe' or  self.args.token_type=='non-overlap':
            self.down_mlp_a = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_t = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_c = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_g = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_n = nn.Linear(config.hidden_size, config.hidden_size)
            self.down_mlp_dict = {
                'A': self.down_mlp_a,
                'T': self.down_mlp_t,
                'C': self.down_mlp_c,
                'G': self.down_mlp_g,
                'N': self.down_mlp_n,
                }
    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)

        if self.args.model_type == 'rnalm':
            hidden_states = output.last_hidden_state
        elif self.args.model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
            hidden_states = output[0]
  
        ## L*ch-> LxL*ch
        batch_size = hidden_states.shape[0]
        weight_mask = data_dict['weight_mask'] #[bz,ori_max_len+2]
        post_token_length = data_dict['post_token_length']


        ### init mappint tensor
        ori_length = weight_mask.shape[1]
        batch_size = hidden_states.shape[0]
        cur_length = int(hidden_states.shape[1])

        if self.args.token_type == 'single':
            #print(attention_mask.shape,weight_mask.shape,post_token_length.shape)
            assert attention_mask.shape==weight_mask.shape==post_token_length.shape
            mapping_hidden_states = hidden_states
        elif self.args.token_type == 'bpe' or  self.args.token_type=='non-overlap':
            inter_input = torch.zeros((batch_size, ori_length, self.config.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
            nucleotide_indices = {nucleotide: (input_ids == self.tokenizer.encode(nucleotide, add_special_tokens=False)[0]).nonzero() for nucleotide in 'ATCGN'}
            mapping_hidden_states = torch.zeros((batch_size, ori_length, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
            for bz in range(batch_size):
                start_index = 0
                for i, length in enumerate(post_token_length[bz]): #astart from [cls]
                    mapping_hidden_states[bz,start_index:start_index + int(length.item()), :] = hidden_states[bz,i,:]
                    start_index += int(length.item())
            for nucleotide, indices in nucleotide_indices.items(): # indices:[bzid,seqid]
     
                if indices.numel() > 0:  
                    bz_indices, pos_indices = indices.split(1, dim=1)
                    bz_indices = bz_indices.squeeze(-1) 
                    pos_indices = pos_indices.squeeze(-1)
                    nucleotide_logits = self.down_mlp_dict[nucleotide](mapping_hidden_states[bz_indices, pos_indices])
                    nucleotide_logits = nucleotide_logits.to(inter_input.dtype)
                    inter_input.index_put_((bz_indices, pos_indices), nucleotide_logits)

        elif 'mer' in self.args.token_type:
            kmer=int(self.args.token_type[0])
            mapping_hidden_states = torch.zeros((batch_size, ori_length, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
            mapping_hidden_states[:,0,:] = hidden_states[:,0,:] #[cls] token
            for bz in range(batch_size):
                value_length = torch.sum(attention_mask[bz,:]==1).item()
                print(value_length)
                for i in range(1,value_length-1): #exclude cls,sep token
                    mapping_hidden_states[bz,i:i+kmer,:] += hidden_states[bz,i]
                mapping_hidden_states[bz,value_length+kmer-1-1,:] = hidden_states[bz,value_length-1,:] #[sep] token
  
        hidden_states = mapping_hidden_states * weight_mask.unsqueeze(2)    
            
        matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)

        matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d

        x = self.cnn(matrix)

        x = x.squeeze(-1)

        return x
