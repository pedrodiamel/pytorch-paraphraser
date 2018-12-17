

import os
import itertools
import random
from utils import (normalizeString, filterPairs )
from language import (Lang, inputVar, outputVar )


def readLangs(pathname, lang1, lang2, reverse=False):
    '''Read language from file 
    Arg:
        pathname: (../rec/data/)
        lang1: (por)
        lang2: (eng)
        reverse: if reverse languaje in the text
    '''
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(  os.path.join(pathname, '{}-{}.txt'.format(lang1, lang2) ) , encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(pathname, lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(pathname, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



class TxtDataset( object ):
    '''TxtDataset
    Args:
        pathname
        lang1
        lang2
        reserse
    '''

    def __init__(self, pathname, lang1, lang2, small_batch_size=5, reserse=True):
        
        self.pathname = pathname
        self.lang1 = lang1
        self.lang2 = lang2
        self.reserse = reserse
        self.small_batch_size = small_batch_size

        #create dataset
        input_lang, output_lang, pairs = prepareData( pathname, lang1, lang2, reserse )
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        

    def getbatch(self):
        return self.batch2TrainData( self.input_lang, self.output_lang,  [random.choice(self.pairs) for _ in range(self.small_batch_size)]  )

    # Returns all items for a given batch of pairs
    def batch2TrainData(self, input_lang, output_lang , pair_batch):
        
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])        
        inp, lengths = inputVar(input_batch, input_lang)
        output, mask, max_target_len = outputVar(output_batch, output_lang)
        return inp, lengths, output, mask, max_target_len




def test_prepare_data():
    input_lang, output_lang, pairs = prepareData('../../rec/data', 'eng', 'por', True)
    # print(random.choice(pairs))
    for pair in pairs[:10]:
        print(pair)

# test_prepare_data()

def test_dataset():

    dataset = TxtDataset( 
        '../../rec/data', 
        'eng', 
        'por', 
        small_batch_size=5,
        reserse=True
    )
    
    batches = dataset.getbatch()
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)


# test_dataset()