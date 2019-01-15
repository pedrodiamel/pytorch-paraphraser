

import sys
sys.path.append('../')
import pytest

from torchlib.datasets import utils
from torchlib.datasets import embeddings

def test_load_sentence_embeddings():    
    pathname='../rec/data/ngram-word-concat-40.pickle'
    emb = embeddings.load_sentence_embeddings( pathname )    
    # print( emb['word2index'] )  
    # print( emb['index2word'] ) 
    # print( emb['word2count'] )
    # print( emb['embeddings'] )
    print( emb['n_words']    )            
    print( emb['index2word'][emb['PAD_token']]  ) 
    print( emb['index2word'][emb['SOS_token']]  ) 
    print( emb['index2word'][emb['EOS_token']]  ) 
    print( emb['index2word'][emb['UNK_token']]  )

test_load_sentence_embeddings()

