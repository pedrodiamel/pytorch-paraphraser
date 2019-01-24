

import os
import numpy as np
import itertools
import random

import torch
import torch.nn as nn

from .utils import (normalizeString, filterPairs, read_paraphraser )
from .vocabulary import (Vocabulary, inputVar, outputVar )
from .downloads import download_data

def prepare_data( pathdataset, pathvocabulary, max_length=10 ):
    
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

def get_triplets( pairs ):
    n = len(pairs)
    i  = np.arange( n )
    j  = np.arange( n )
    ij = np.array([0,1]); random.shuffle( ij ) 
    while np.sum( (np.abs(i-j) == 0 ) ) != 0:
        random.shuffle( j )
    triplets = [ ((pairs[i][ ij[0] ], pairs[i][ ij[1] ], pairs[j[i]][ random.randint( 0,1 ) ]))  for i in range(n) ]    
    return triplets

def get_pairs( dataset, n=5 ):    
    pairs = [] 
    Cls, Words = dataset[:,0], dataset[:,1]
    C,F = np.unique( Cls, return_counts=True )
    for c,f in zip(C,F):         
        # a, b   
        a = np.array( np.random.choice( np.where(Cls==c)[0], min(f,n), replace=False ))
        b = np.array( np.random.choice( np.where(Cls==c)[0], min(f,n), replace=False ))
        #while np.any((a-b)==0): #aligning check
        while np.sum((a-b) == 0 )/b.shape[0] > 0.1: #aligning check
            random.shuffle(b) 
        pairs += zip(Words[a],Words[b])
        #print(c, a,b)
    random.shuffle(pairs)
    return pairs




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
        max_length=10
        ):
        self.pathname       = os.path.expanduser( pathname )
        self.filevocabulary = filevocabulary
        self.filedataset    = filedataset
        self.pathvocabulary = os.path.join( self.pathname, filevocabulary )
        self.pathdataset    = os.path.join( self.pathname, filedataset )

        #if not os.path.exists( self.pathname ):
        #    download_data( self.namefile, self.idfile, self.pathname, ext=True )
        
        #create dataset
        voc, pairs = prepare_data( self.pathdataset, self.pathvocabulary, max_length )
        self.voc = voc
        self.pairs = pairs
        self.batch_size = batch_size if batch_size else len(pairs)
        self.nbatch = nbatch

    def __len__(self):
        return self.nbatch #self.batch_size*

class TxtTripletDataset( TxtDataset ):
    '''TxtTripletDataset
    '''
    def __init__(self, 
        pathname, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10
        ):
        super(TxtTripletDataset, self).__init__(  pathname, filedataset, filevocabulary, nbatch, batch_size, max_length)


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
        max_length=10
        ):
        super(TxtPairDataset, self).__init__(  pathname, filedataset, filevocabulary, nbatch, batch_size, max_length)

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
        ):
        super(TxtNMTDataset, self).__init__(  pathname, filedataset, filevocabulary, nbatch, batch_size, max_length)


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

