
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
    UNK = voc.index2word[voc.UNK_token]     
    decoded_words = [voc.index2word.get(token.item(), UNK) for token in tokens] 
    return decoded_words



def evaluateInput(net, voc, max_length):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('>> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            decoded_words = evaluate(net, voc, input_sentence, max_length  )
            # Format and print response sentence            
            EOS = voc.index2word[voc.EOS_token]
            PAD = voc.index2word[voc.PAD_token]
            decoded_words[:] = [x for x in decoded_words if not (x == EOS or x == PAD)]
            print('Generate:', ' '.join(decoded_words))
        except KeyError:
            print("Error: Encountered unknown word.")




# def arg_parser():
#     """Arg parser"""    
#     parser = ArgumentParser()
#     parser.add_argument('data', metavar='DIR', help='paraphraser')
#     return parser

def main():
 
    pathmodel      = 'out/netruns/nlp_nmt_maskll_adam_txt_000/models/model_best.pth.tar'
    pathvocabulary = '~/.datasets/txt/para-nmt-50m-demo/ngram-word-concat-40.pickle'    
    max_length     = 10
    parallel       = False
    no_cuda        = True
    gpu            = 0
    
    sentence       = 'this is a difficult test !'

    # load vocabulary
    print('>> Load vocabulary ...')
    pathvocabulary = os.path.expanduser( pathvocabulary )
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


#     # evaluate
#     sentence = normalizeString(sentence)
#     decoded_words = evaluate(network, voc, sentence, max_length  )    
#     EOS = voc.index2word[voc.EOS_token]
#     PAD = voc.index2word[voc.PAD_token]
#     decoded_words[:] = [x for x in decoded_words if not (x == EOS or x == PAD)]
#     print('>> REUSLT: ')
#     print('>> input: ', sentence)
#     print('<< output: ',  ' '.join(decoded_words) )


    evaluateInput(network, voc, max_length)



if __name__ == '__main__':
    main()
