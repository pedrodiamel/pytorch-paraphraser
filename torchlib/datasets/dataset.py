

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
    i = np.arange( n )
    j = np.arange( n )
    while np.sum( (np.abs(i-j) == 0 ) ) != 0:
        random.shuffle( j )
    triplets = [ ((pairs[i][0], pairs[i][1], pairs[j[i]][ random.randint( 0,1 ) ]))  for i in range(n) ]    
    return triplets


class TxtTripletDataset( object ):
    '''TxtTripletDataset
    Args:
        pathname
        pathvocabulary
        nbatch
        batch_size
    '''

    def __init__(self, pathname, pathvocabulary, nbatch=100, batch_size=5 ):
        self.pathname = pathname
        self.pathvocabulary = pathvocabulary
        self.batch_size = batch_size
        self.nbatch = nbatch        
        #create dataset
        voc, pairs = prepare_data( pathname, pathvocabulary )
        self.voc = voc
        self.pairs = pairs

    def __len__(self):
        return self.nbatch

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.pairs) for _ in range(self.batch_size)]  )

    def getbatchs(self):
        for _ in range( self.nbatch ):
            yield self.getbatch()

    # Returns all items for a given batch of triplet
    def batch2TrainData(self, pair_batch):  
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        triple_batch = get_triplets(pair_batch)
        s1_batch, s2_batch, t1_batch = [], [], []
        for triple in triple_batch :
            s1_batch.append(triple[0])
            s2_batch.append(triple[1])
            t1_batch.append(triple[2])  
        s1, s1_mask, s1_max_len = outputVar(s1_batch, self.voc)
        s2, s2_mask, s2_max_len = outputVar(s2_batch, self.voc)
        t1, t1_mask, t1_max_len = outputVar(t1_batch, self.voc)    
        return (
            s1, s1_mask, s1_max_len, 
            s2, s2_mask, s2_max_len, 
            t1, t1_mask, t1_max_len 
            )
