

import os
import random 
import numpy as np

import torch
import torch.nn.functional as F

from torchlib.datasets.dataset  import TxtNMTDataset
from torchlib.nmtneuralnet import NeuralNetNMT
from torchlib.datasets.vocabulary import ( indexesFromSentence )
from torchlib.datasets.utils import ( normalizeString, filterPairs )


def main():
    
    # configurate 
    pathname       = '~/.datasets/txt'
    namedataset    = 'cmds' #cmds, paranmt,
    pathdata       = 'dbcommand.csv' #dbcommand.csv; commandpairsext.txt; para-nmt-50m/para-nmt-50m.txt; para-nmt-50m-demo/para-nmt-50m-small.txt 
    pathvocabulary = 'para-nmt-50m-demo/ngram-word-concat-40.pickle'
    pathmodel      = 'out/netruns/nlp_nmt_maskll_adam_paranmt_004/models/model_best.pth.tar'
    nbatch         = 50 
    batch_size     = 100
    max_length     = 10
    no_cuda        = False
    seed           = 0
    gpu            = 0
    parallel       = False
    
    # load dataset
    dataset = TxtNMTDataset(
        pathname=pathname,
        namedataset=namedataset,
        filedataset=pathdata,
        filevocabulary=pathvocabulary,
        nbatch=nbatch,
        batch_size=batch_size,
        max_length=max_length,
    )
        
    print('Dataset')
    print('Size: ', len(dataset))

    # load model 
    network = NeuralNetNMT(
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        gpu=gpu
        )

    if network.load( pathmodel ) is not True:
        raise ValueError('Error: model not load ...')
    
    print( 'load NeuralNet ... ' )
    print( network )   

    bleu = network.test( dataset )
    print('BLEU: ', bleu)
    print('DONE!!!!')
   


if __name__ == '__main__':
    main()

