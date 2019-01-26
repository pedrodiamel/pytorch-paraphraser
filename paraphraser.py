
import os 
import torch

from torchlib.datasets.vocabulary import ( Vocabulary, indexesFromSentence )
from torchlib.datasets.utils import ( normalizeString )
from torchlib.nmtneuralnet import NeuralNetNMT
from argparse import ArgumentParser


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
    decoded_words = [voc.index2word.get(token.item(), voc.index2word[ voc.UNK_token] ) for token in tokens] 
    return decoded_words

# def arg_parser():
#     """Arg parser"""    
#     parser = ArgumentParser()
#     parser.add_argument('data', metavar='DIR', help='paraphraser')
#     return parser

def main():

    gpu            = 0
    no_cuda        = True
    parallel       = False
    pathmodel      = 'out/netruns/nlp_nmt_maskll_adam_paranmt_003/models/model_best.pth.tar'
    pathvocabulary = '~/.datasets/txt/para-nmt-50m-demo/ngram-word-concat-40.pickle'
    pathvocabulary = os.path.expanduser( pathvocabulary )    
    paraphraser    = 'This a test !'


    # load vocabulary
    print('>> Load vocabulary ...')
    voc = Vocabulary()
    voc.load_embeddings( pathvocabulary, type='emb' ) 
    print(">> Counted words:")
    print(voc.n_words)

    # load model 
    print('>> Load model ...')
    network = NeuralNetNMT(
        no_cuda=no_cuda,
        parallel=parallel,
        gpu=gpu
        )

    if network.load( pathmodel, voc ) is not True:
        raise ValueError('Error: model not load ...')
    print( network )


    # evaluate
    decoded_words = evaluate(network, voc, paraphraser,  )

    print('<< REUSLT: ')
    print('<< ',  decoded_words)



if __name__ == '__main__':
    main()
