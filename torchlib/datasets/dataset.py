

import os
import numpy as np
import itertools
import random

import torch
import torch.nn as nn

from .utils import ( 
    normalizeString,  
    get_triplets, 
    txtParaNmt50mProvide,
    txtCmdPairsProvide
    )
from .vocabulary import (Vocabulary, inputVar, outputVar )


class TxtDataset( object ):
    '''TxtDataset
    Args:
        pathname
        namedataset
        filedataset
        filevocabulary
        nbatch
        batch_size
        max_length
    '''
    
    def __init__(self, 
        pathname, 
        namedataset,
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10, 
        ):
        self.pathname       = os.path.expanduser( pathname )
        self.namedataset    = namedataset
        self.filevocabulary = filevocabulary
        self.filedataset    = filedataset
        self.pathvocabulary = os.path.join( self.pathname, filevocabulary )
        self.pathdataset    = os.path.join( self.pathname, filedataset )


        data = None
        if namedataset == 'paranmt':
            data = txtParaNmt50mProvide( self.pathdataset )
            data.filter( max_length )
        elif namedataset == 'cmds':
            data = txtCmdPairsProvide( self.pathdataset )
            data.filter( max_length )
        else:
            raise ValueError('Name dataset not sopport.')

        #create vocabulary
        voc = Vocabulary()
        #load vocabulary
        voc.load_embeddings( self.pathvocabulary, type='emb' ) 
        print("Counted words:")
        print(voc.n_words)
        
         

        self.voc = voc
        self.data = data
        self.batch_size = batch_size if batch_size else len(data)
        self.nbatch = nbatch

    def __len__(self):
        return self.nbatch #self.batch_size




class TxtTripletDataset( TxtDataset ):
    '''TxtTripletDataset
    '''
    def __init__(self, 
        pathname, 
        namedataset,
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        ):
        super(TxtTripletDataset, self).__init__(  pathname, namedataset, filedataset, filevocabulary, nbatch, batch_size, max_length )

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.data.pairs) for _ in range(self.batch_size)]  )

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
        namedataset,
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,         
        ):
        super(TxtPairDataset, self).__init__(  pathname, namedataset, filedataset, filevocabulary, nbatch, batch_size, max_length )

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        pair = self.data[ i%len(self.data) ]
        s1, s1_mask, s1_max_len = outputVar([pair[0]], self.voc)
        s2, s2_mask, s2_max_len = outputVar([pair[1]], self.voc) 
        return (
            pair[0], s1, s1_mask, s1_max_len, 
            pair[1], s2, s2_mask, s2_max_len, 
            )       

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.data.pairs) for _ in range(self.batch_size)]  )

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
        namedataset, 
        filedataset,
        filevocabulary, 
        nbatch=100, 
        batch_size=None,
        max_length=10,
        ):
        super(TxtNMTDataset, self).__init__( pathname, namedataset, filedataset, filevocabulary, nbatch, batch_size, max_length)


    def __len__(self):
        return self.nbatch #self.batch_size*

    def __getitem__(self, i):
        pair = self.data[ i%len(self.data) ]
        inp, lengths = inputVar([pair[0]], self.voc)
        output, mask, max_target_len = outputVar([pair[1]], self.voc)
        return (
            inp, lengths,  
            output, mask, max_target_len, 
            )       

    def getbatch(self):
        return self.batch2TrainData( [random.choice(self.data.pairs) for _ in range(self.batch_size)]  )

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

