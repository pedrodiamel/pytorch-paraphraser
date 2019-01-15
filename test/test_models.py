

import sys
sys.path.append('../')
import pytest


import torch
import torch.nn as nn

from torchlib.datasets import dataset
from torchlib.models import  tripletnet


def test_model( type='avg'):

    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtTripletDataset( 
        pathname,  
        pathvocabulary,
        batch_size=5,
        nbatch=10,
    )
    
    batches = data.getbatch()
    s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batches
    embedding = nn.Embedding.from_pretrained( torch.from_numpy( data.voc.embeddings ).float() )
    encmodel=[]
    
    if type == 'avg':
        encmodel = tripletnet.EncoderAvg( embedding )
    elif type == 'rnn':
        encmodel = tripletnet.EncoderRNN( embedding=embedding, hidden_size=300, n_layers=1 )
    else:
        print('[Error]: Type not soport !!!!')
        assert(False)

    tripmodel = tripletnet.Tripletnet( encmodel )

    print( s1.shape )
    print( s1_mask.shape )

    s1_ave_enc, s2_ave_enc, t1_ave_enc = tripmodel( s1, s1_mask, s2, s2_mask, t1, t1_mask )

    print( s1_ave_enc.shape )
    print( s2_ave_enc.shape )
    print( t1_ave_enc.shape )



test_model( type='rnn' )
