

import sys
sys.path.append('../')
import pytest

import torch
import torch.nn as nn

from torchlib.datasets import dataset
from torchlib.models import  tripletnet
from torchlib.models import  attnet
from torchlib import loss as netloss

def test_triplet_model( type='avg'):

    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtTripletDataset( 
        pathname,  
        pathvocabulary,
        batch_size=5,
        nbatch=10,
    )
    
    hidden_size=300
    batches = data.getbatch()
    s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batches
    #embedding = nn.Embedding.from_pretrained( torch.from_numpy( data.voc.embeddings ).float() )
    embedding = nn.Embedding( data.voc.n_words, hidden_size )

    encmodel=[]
    
    if type == 'avg':
        encmodel = tripletnet.EncoderAvg( embedding )
    elif type == 'rnn':
        encmodel = tripletnet.EncoderRNNAvg( hidden_size=hidden_size, embedding=embedding, n_layers=1, dropout=0, tonorm=True )
       
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



def test_nmt_model( ):
    
    # configurate 
    batch_size=5
    attn_model = 'dot'
    hidden_size = 300
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1

    # datasets
    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtNMTDataset( 
        pathname,  
        pathvocabulary,
        batch_size=batch_size,
        nbatch=10,
    )
    
    batches = data.getbatch()
    inp, lengths, output, mask, max_target_len,  = batches

    print()
    print( 'inp shape: ', inp.shape )
    print( 'output shape: ', output.shape )
    print( 'mask shape: ', mask.shape )
    print()
    
    # models
    
    
    dim=data.voc.n_words #74666 #74602 #data.voc.n_words
    #embedding = nn.Embedding( dim, hidden_size )
    embedding = nn.Embedding.from_pretrained( torch.from_numpy( data.voc.embeddings ).float() )
    encoder   = attnet.EncoderRNN( hidden_size, embedding, encoder_n_layers, dropout )
    decoder   = attnet.LuongAttnDecoderRNN( attn_model, embedding, hidden_size, dim, decoder_n_layers, dropout )

    # evaluate
    encoder_outputs, encoder_hidden = encoder( inp, lengths )
    
    print( 'encoder shape: ', encoder_outputs.shape )
    print( 'encoder hidden shape: ', encoder_hidden.shape )
    print()

    decoder_input = torch.LongTensor([[data.voc.SOS_token for _ in range(batch_size)]])
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    print('decoder_input shape: ', decoder_input.shape)
    print('decoder_hidden shape: ', decoder_hidden.shape)
    print()
  
    loss = 0
    n_totals = 0

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden, encoder_outputs )
        # Teacher forcing: next input is current target
        decoder_input = output[t].view(1, -1)
        # Calculate and accumulate loss
        mask_loss, nTotal = netloss.MaskNLLLoss()(decoder_output, output[t], mask[t] )
        loss += mask_loss
        n_totals += nTotal

        #print('>> ', t)

    print( 'mask loss: ', loss )
    print( 'nTotal: ', n_totals )
    print('DONE!!!')    





test_triplet_model( type='rnn' )
# test_nmt_model()