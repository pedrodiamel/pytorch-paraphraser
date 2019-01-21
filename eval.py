

import os
import random 
import numpy as np

import torch
import torch.nn.functional as F

from torchlib.datasets.dataset  import TxtNMTDataset
from torchlib.nmtneuralnet import NeuralNetNMT

from torchlib.datasets.vocabulary import ( indexesFromSentence )
from torchlib.datasets.utils import ( normalizeString, filterPairs )


def evaluate(net, voc, sentence, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]    
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)  
      
    # Decode sentence with searcher
    tokens, scores = net(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateRandomly(net, voc, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        input_sentence = normalizeString( pair[0] )
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(net, voc, input_sentence, max_length=len(input_sentence)+1 )
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')




def main():
    
    # configurate 
    pathdata = './rec/data/para-nmt-50m-small.txt'
    pathvocabulary = './rec/data/ngram-word-concat-40.pickle'
    nbatch=5
    batch_size=10

    project='./out/netruns'
    name='nlp_nmt_maskll_adam_txt_003'
    no_cuda=True
    seed=0
    gpu=0
    parallel=False
    pathmodel=os.path.join( project, name, 'models/model_best.pth.tar' )

    # load dataset
    dataset = TxtNMTDataset(
        pathname=pathdata,  
        pathvocabulary=pathvocabulary,
        nbatch=nbatch,
        batch_size=batch_size
    )

    print('Dataset')
    print('Size: ', len(dataset))

    # load model 
    network = NeuralNetNMT(
        patchproject=project,
        nameproject=name,
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
    # evaluateRandomly( network, dataset.voc, dataset.pairs )



if __name__ == '__main__':
    main()

