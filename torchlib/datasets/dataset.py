

import os
import numpy as np
import itertools
import random

import torch
import torch.nn as nn

from .utils import (normalizeString, filterPairs, read_paraphraser )
from .vocabulary import (Vocabulary, inputVar, outputVar )


def prepare_data( pathname, pathvocabulary ):
    pairs = read_paraphraser( pathname)
    voc = Vocabulary()
    voc.load_embeddings( pathvocabulary, type='emb' )  
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)        
    print("Counted words:")
    print(voc.n_words)
    return voc, pairs

def get_triplets( pairs ):
    n = len(pairs)
    i  = np.arange( n )
    j  = np.arange( n )
    ij = np.array([0,1]); random.shuffle( ij ) 
    while np.sum( (np.abs(i-j) == 0 ) ) != 0:
        random.shuffle( j )
    triplets = [ ((pairs[i][ ij[0] ], pairs[i][ ij[1] ], pairs[j[i]][ random.randint( 0,1 ) ]))  for i in range(n) ]    
    return triplets


class TxtTripletDataset( object ):
    '''TxtTripletDataset
    Args:
        pathname
        pathvocabulary
        nbatch
        batch_size
    '''

    def __init__(self, 
        pathname, 
        pathvocabulary, 
        nbatch=100, 
        batch_size=None 
        ):

        self.pathname = pathname
        self.pathvocabulary = pathvocabulary              
        #create dataset
        voc, pairs = prepare_data( pathname, pathvocabulary )
        self.voc = voc
        self.pairs = pairs
        self.batch_size = batch_size if batch_size else len(pairs)
        self.nbatch = nbatch  

    def __len__(self):
        return self.nbatch

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.pairs) for _ in range(self.batch_size)]  )

    def getbatchs(self):
        for _ in range( self.nbatch ):
            yield self.getbatch()

    # Returns all items for a given batch of triplet
    def batch2TrainData(self, pair_batch):  

        #pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

        triple_batch = get_triplets(pair_batch)
        s1_batch, s2_batch, t1_batch = [], [], []
        for triple in triple_batch :
            s1_batch.append(triple[0])
            s2_batch.append(triple[1])
            t1_batch.append(triple[2])  
       

        s1, s1_mask, s1_max_len = outputVar(s1_batch, self.voc)
        s2, s2_mask, s2_max_len = outputVar(s2_batch, self.voc)
        t1, t1_mask, t1_max_len = outputVar(t1_batch, self.voc)    

        # s1_index = s1_mask.sum(axis=0).argsort() 
        # s2_index = s2_mask.sum(axis=0).argsort() 
        # t1_index = t1_mask.sum(axis=0).argsort()        

        return (
            s1, s1_mask, s1_max_len, 
            s2, s2_mask, s2_max_len, 
            t1, t1_mask, t1_max_len 
            )



class TxtPairDataset( object ):
    '''TxtPairDataset
    Args:
        pathname
        pathvocabulary
        nbatch
        batch_size
    '''

    def __init__(self, 
        pathname, 
        pathvocabulary, 
        nbatch=100, 
        batch_size=None 
        ):
        self.pathname = pathname
        self.pathvocabulary = pathvocabulary
              
        #create dataset
        voc, pairs = prepare_data( pathname, pathvocabulary )
        self.voc = voc
        self.pairs = pairs

        self.batch_size = batch_size if batch_size else len(pairs)
        self.nbatch = nbatch  

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        pair = self.pairs[ i%len(self.pairs) ]
        s1, s1_mask, s1_max_len = outputVar([pair[0]], self.voc)
        s2, s2_mask, s2_max_len = outputVar([pair[1]], self.voc) 
        return (
            pair[0], s1, s1_mask, s1_max_len, 
            pair[1], s2, s2_mask, s2_max_len, 
            )       

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.pairs) for _ in range(self.batch_size)]  )

    def getbatchs(self):
        for _ in range( self.nbatch ):
            yield self.getbatch()

    # Returns all items for a given batch of triplet
    def batch2TrainData(self, pair_batch):  
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)        
        s1_batch, s2_batch = [], []
        for pair in pair_batch :
            s1_batch.append(pair[0])
            s2_batch.append(pair[1])            
        s1, s1_mask, s1_max_len = outputVar(s1_batch, self.voc)
        s2, s2_mask, s2_max_len = outputVar(s2_batch, self.voc)          
        return (
            s1, s1_mask, s1_max_len, 
            s2, s2_mask, s2_max_len, 
            )
