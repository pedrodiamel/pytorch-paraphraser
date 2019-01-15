

import torch
import itertools
from . import embeddings


class Vocabulary:

    def __init__(self ):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        self.embeddings = []        
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token
        self.UNK_token = 3  # Unkown sentence token
        

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def load_embeddings( self, pathname, type='emb' ):

        if type == 'emb':    
            emb = embeddings.load_sentence_embeddings(pathname)
            self.word2index = emb['word2index']  
            self.index2word = emb['index2word']  
            self.word2count = emb['word2count']  
            self.embeddings = emb['embeddings']  
            self.n_words    = emb['n_words']             
            self.PAD_token  = emb['PAD_token']  
            self.SOS_token  = emb['SOS_token']  
            self.EOS_token  = emb['EOS_token']  
            self.UNK_token  = emb['UNK_token']
        else: 
            print('Not embedding load type ...')
            assert(False)





def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, voc.UNK_token )  for word in sentence.split(' ')] + [ voc.EOS_token ]

def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, voc.EOS_token)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, voc.EOS_token)
    mask = binaryMatrix(padList, voc.EOS_token)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


