
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

__all__ = ['Tripletnet', 'EncoderAvg', 'EncoderRNN', 'encoder_ave', 'encoder_rnn']
    
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
    
    def forward(self, s1, s1_mask, s2, s2_mask, t1, t1_mask):
        embedded_s1 = self.embeddingnet(s1, s1_mask)
        embedded_s2 = self.embeddingnet(s2, s2_mask)
        embedded_t1 = self.embeddingnet(t1, t1_mask)
        return embedded_s1, embedded_s2, embedded_t1


class WieghtAverage(nn.Module):
    '''WieghtAverage'''
    def __init__(self, tonorm=True):
        super(WieghtAverage, self).__init__()      
        self.tonorm = tonorm
    def forward(self, x, w ):                       
        #Create average embedding vector
        y = (x * w.view( w.shape[0],w.shape[1],-1 ).float())  
        y = y.sum( dim=0 )
        #if normalization 
        if self.tonorm: 
            y = y / w.float().sum(dim=0).unsqueeze(dim=1)  
        return y


class EncoderAvg(nn.Module):
    '''Encoder average'''
    def __init__(self, embedding, tonorm=True ):
        super(EncoderAvg, self).__init__()
        self.embedding = embedding      
        self.avg = WieghtAverage( tonorm )

    def forward(self, input_seq, input_mask ):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq) 
        outputs = self.avg( embedded, input_mask )
        return outputs

  
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, tonorm=True):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.avg = WieghtAverage( tonorm )

    def forward(self, input_seq, input_mask, hidden=None):        
        # Calculate length
        input_lengths = input_mask.sum(dim=0) + 1
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        outputs = self.avg( outputs, input_mask )

        print('DONE!!!')
        assert(False)

        return outputs


def encoder_ave(pretrained=False, **kwargs):
    """Average"""
    model = EncoderAvg(**kwargs)
    if pretrained:
        pass
    return model

def encoder_rnn(pretrained=False, **kwargs):
    """RNN"""
    model = EncoderRNN(hidden_size=300, n_layers=2, **kwargs)
    if pretrained:
        pass
    return model