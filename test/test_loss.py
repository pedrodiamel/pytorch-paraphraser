

import sys
sys.path.append('../')
import pytest

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import torch
import torch.nn as nn
from torchlib.models import  attnet
from torchlib import loss as netloss
from torchlib.datasets import dataset

def test_triplet_loss():

    dim=300; batch=5
    s1 = torch.randn( batch, dim )
    s2 = torch.randn( batch, dim )
    t1 = torch.randn( batch, dim )

    print(s1.shape)
    print(s2.shape)
    print(t1.shape)

    l = netloss.tripletCosineLoss()(s1, s2, t1)
    print(l)


def test_maskll_loss():
    
    dim=300; 
    batch=5   
    inp = torch.rand( batch, dim )
    out = (torch.randint(0,10, (batch,1) ) ).long() 
    mask = (torch.rand( batch, 1 ) > 0.5 ).byte()

    print(inp.shape)
    print(out.shape)
    print(mask.shape)

    l = netloss.MaskNLLLoss()(inp, out, mask)
    print(l)







def test_bleu():
    
    hyp1  = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
    hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history']
    ref2a = ['he', 'was', 'interested', 'in', 'world', 'history', 'because', 'he', 'read', 'the', 'book']
    
    list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    
    blue = corpus_bleu(list_of_references, hypotheses)  
    print(blue)


def test_metric_bleu( ):
    
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
    embedding = nn.Embedding( dim, hidden_size )
    #embedding = nn.Embedding.from_pretrained( torch.from_numpy( data.voc.embeddings ).float() )

    encoder = attnet.EncoderRNN( hidden_size, embedding, encoder_n_layers, dropout )
    decoder = attnet.LuongAttnDecoderRNN( attn_model, embedding, hidden_size, dim, decoder_n_layers, dropout )
    search  = attnet.GreedySearchDecoder( encoder, decoder, sos=data.voc.SOS_token, cuda=False )

    # evaluate
  
    tokens_batch, scores_batch = search( inp, lengths, max_target_len, batch_size )

    print( tokens_batch.shape )
    #print( tokens_batch )
    print( scores_batch.shape ) 
    #print( scores_batch )


    all_hyp_words = []
    all_ref_words = []
    for j in range(batch_size): 
        tokens_hyp = tokens_batch[:,j]
        tokens_ref = output[:,j]

        decoded_hyp_words = [data.voc.index2word[ token.item() ] for token in tokens_hyp]
        decoded_hyp_words[:] = [x for x in decoded_hyp_words if not (x == data.voc.EOS_token or x == data.voc.PAD_token)]
        all_hyp_words.append( decoded_hyp_words )
        
        decoded_ref_words = [data.voc.index2word[ token.item() ] for token in tokens_ref]
        decoded_ref_words[:] = [x for x in decoded_ref_words if not (x == data.voc.EOS_token or x == data.voc.PAD_token)]
        all_ref_words.append( decoded_ref_words )

    print( all_hyp_words[0] )
    print( all_ref_words[0] )

    blue = corpus_bleu(all_ref_words, all_hyp_words) 
    print('Bleu: ', blue)

# test_triplet_loss()
# test_maskll_loss()
# test_bleu()
test_metric_bleu()