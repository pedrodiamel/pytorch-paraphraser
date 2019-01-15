
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

__all__ = ['Tripletnet', 'EncoderAvg', 'encoder_ave']
    
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
    
    def forward(self, s1, s1_mask, s2, s2_mask, t1, t1_mask):
        embedded_s1 = self.embeddingnet(s1, s1_mask)
        embedded_s2 = self.embeddingnet(s2, s2_mask)
        embedded_t1 = self.embeddingnet(t1, t1_mask)
        return embedded_s1, embedded_s2, embedded_t1


class EncoderAvg(nn.Module):
    '''Encoder average'''
    def __init__(self, embedding, tonorm=True):
        super(EncoderAvg, self).__init__()
        self.embedding = embedding      
        self.tonorm = tonorm

    def forward(self, input_seq, input_mask ):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)        
        #Create average embedding vector
        outputs = (embedded * input_mask.view( input_mask.shape[0],input_mask.shape[1],-1 ).float())  
        outputs = outputs.sum( dim=0 )
        #if normalization 
        if self.tonorm: 
            outputs = outputs / input_mask.float().sum(dim=0).unsqueeze(dim=1)        
        return outputs
    

def encoder_ave(pretrained=False, **kwargs):
    """Average"""
    model = EncoderAvg(**kwargs)
    if pretrained:
        pass
    return model

