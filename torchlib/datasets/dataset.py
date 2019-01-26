

import os
import numpy as np
import itertools
import random

import torch
import torch.nn as nn

from .utils import (normalizeString, filterPairs, get_triplets, read_pairs, read_words, read_word2pairs )
from .vocabulary import (Vocabulary, inputVar, outputVar )
from .downloads import download_data

def prepare_data( pathdataset, pathvocabulary, read_paraphraser=read_pairs, max_length=10 ):
    
    #read dataset
    pairs = read_paraphraser( pathdataset )
    #create vocabulary
    voc = Vocabulary()
    #load vocabulary
    voc.load_embeddings( pathvocabulary, type='emb' )  
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length=max_length)        
    print("Counted words:")
    print(voc.n_words)
    return voc, pairs


class TxtDataset( object ):
    '''TxtDataset
    Args:
        pathname
        filedataset
        filevocabulary
        nbatch
        batch_size
        max_length
    '''
    
    idfile = '1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD'
    namefile = 'para-nmt-50m-demo.zip'

    def __init__(self, 
        pathname, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        read_paraphraser=read_pairs,
        ):
        self.pathname       = os.path.expanduser( pathname )
        self.filevocabulary = filevocabulary
        self.filedataset    = filedataset
        self.pathvocabulary = os.path.join( self.pathname, filevocabulary )
        self.pathdataset    = os.path.join( self.pathname, filedataset )

        #if not os.path.exists( self.pathname ):
        #    download_data( self.namefile, self.idfile, self.pathname, ext=True )
        
        #create dataset
        voc, pairs = prepare_data( self.pathdataset, self.pathvocabulary, read_paraphraser, max_length )

        self.voc = voc
        self.pairs = pairs
        self.batch_size = batch_size if batch_size else len(pairs)
        self.nbatch = nbatch

    def __len__(self):
        return self.nbatch #self.batch_size


class TxtTripletDataset( TxtDataset ):
    '''TxtTripletDataset
    '''
    def __init__(self, 
        pathname, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        read_paraphraser=read_pairs,
        ):
        super(TxtTripletDataset, self).__init__(  pathname, filedataset, filevocabulary, nbatch, batch_size, max_length, read_paraphraser)


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

class TxtPairDataset( TxtDataset ):
    '''TxtPairDataset
    '''
    def __init__(self, 
        pathname, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        read_paraphraser=read_pairs,
        ):
        super(TxtPairDataset, self).__init__(  pathname, filedataset, filevocabulary, nbatch, batch_size, max_length, read_paraphraser)

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

class TxtNMTDataset( TxtDataset ):
    '''TxtNMTDataset
    '''
    def __init__(self, 
        pathname, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        read_paraphraser=read_pairs
        ):
        super(TxtNMTDataset, self).__init__( pathname, filedataset, filevocabulary, nbatch, batch_size, max_length, read_paraphraser)


    def __len__(self):
        return self.nbatch #self.batch_size*

    def __getitem__(self, i):
        pair = self.pairs[ i%len(self.pairs) ]
        inp, lengths = inputVar([pair[0]], self.voc)
        output, mask, max_target_len = outputVar([pair[1]], self.voc)
        return (
            inp, lengths,  
            output, mask, max_target_len, 
            )       

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.pairs) for _ in range(self.batch_size)]  )

    def getbatchs(self):
        for _ in range( self.nbatch ):
            yield self.getbatch()

    # Returns all items for a given batch of triplet
    def batch2TrainData(self, pair_batch):  
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)        
        in_batch, out_batch = [], []
        for pair in pair_batch :
            in_batch.append(pair[0])
            out_batch.append(pair[1])                    
        inp, lengths = inputVar(in_batch, self.voc)
        output, mask, max_target_len = outputVar(out_batch, self.voc)
        return (
            inp, lengths, 
            output, mask, max_target_len, 
            )

