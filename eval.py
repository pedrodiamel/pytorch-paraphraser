

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
    pathname='~/.datasets/txt'
    namedataset='paranmt'
    pathdata = 'data/para-nmt-50m-small.txt'
    pathvocabulary = 'data/ngram-word-concat-40.pickle'
    nbatch=5
    batch_size=10
    max_length=10

    project='./out/netruns'
    name='nlp_nmt_maskll_adam_txt_003'
    no_cuda=True
    seed=0
    gpu=0
    parallel=False
    pathmodel=os.path.join( project, name, 'models/model_best.pth.tar' )

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

    if network.load( pathmodel, dataset.voc ) is not True:
        raise ValueError('Error: model not load ...')
    
    print( 'load NeuralNet ... ' )
    print( network )   

    bleu = network.test( dataset )
    print('BLEU: ', bleu)
    print('DONE!!!!')
   


if __name__ == '__main__':
    main()

