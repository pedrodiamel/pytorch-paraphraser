
import os
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pandas as pd


# Turn a Unicode string to plain ASCII, thunicodedata
# http://stackoverflow.com/a/518232/280942unicodedata
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPhrase(p, max_length=10 ):
    return len(p.split(' ')) < max_length

def filterPair(p, max_length=10 ):
    return filterPhrase(p[0],max_length) and \
        filterPhrase(p[1],max_length) 

def filterPhrases(phrases, max_length=10):
    return [phrase for phrase in phrases if filterPhrase(phrase, max_length)]

def filterPairs(pairs, max_length=10):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def get_triplets( pairs ):
    n = len(pairs)
    i  = np.arange( n )
    j  = np.arange( n )
    ij = np.array([0,1]); random.shuffle( ij ) 
    while np.sum( (np.abs(i-j) == 0 ) ) != 0:
        random.shuffle( j )
    triplets = [ ((pairs[i][ ij[0] ], pairs[i][ ij[1] ], pairs[j[i]][ random.randint( 0,1 ) ]))  for i in range(n) ]    
    return triplets

def get_pairs( phrases, classes, n=5 ):  
    '''get pairs 
    Args:
        phrases: list of phrases 
        types: list of words types 
        n: represent the number of combination for each class 
    '''  


    pairs = [] 
    C,F = np.unique( classes, return_counts=True )
    for c,f in zip(C,F):         
        # a, b   
        a = np.array( np.random.choice( np.where(classes==c)[0], min(f,n), replace=False ))
        b = np.array( np.random.choice( np.where(classes==c)[0], min(f,n), replace=False ))
        #while np.any((a-b)==0): #aligning check
        while np.sum((a-b) == 0 )/b.shape[0] > 0.1: #aligning check
            random.shuffle(b) 
        pairs += zip(phrases[a],phrases[b])
        #print(c, a,b)
    random.shuffle(pairs)
    return pairs


# TODO January 26, 2019: Include dowload dataset
class txtParaNmt50mProvide( object ):
    '''text ParaNmt50m provide dataset
    Args:
        pathname
    '''
    idfile = '1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD'
    namefile = 'para-nmt-50m-demo.zip'

    def __init__(self,  
        pathname
        ):
        self.pathname = pathname      

        #if not os.path.exists( self.pathname ):
        #    download_data( self.namefile, self.idfile, self.pathname, ext=True )

        #read dataset
        self.pairs = self.load( pathname )
                  
    def __len__(self):
        return len( self.pairs )
    def __getitem__(self, i):
        return self.pairs(i)
    
    def load( self, pathname ):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(pathname, encoding='utf-8').\
            read().strip().split('\n')    
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]    
        return pairs
    def filter(self, max_length=10):
        self.pairs = filterPairs( self.pairs, max_length )

class txtCmdProvide( object ):
    '''text cmd provide dataset
    Args:
        pathname
    '''
    def __init__(self, 
        pathname, 
        filter=True,
        ):
        self.pathname = pathname      
        #read dataset
        self.phrases, self.classes = self.load( pathname )
    def __len__(self):
        return len( self.phrases )
    def __getitem__(self, i):
        return self.phrases[i], self.classes[i]
    def load(self, pathname ):
        data = pd.read_csv( pathname )
        phrases = data['phrases']
        classes = data['classes']
        phrases = np.array([normalizeString(phrase) for phrase in phrases])
        return phrases, classes

    def filter(self, max_length=10):
        self.phrases = filterPhrases(self.phrases, max_length)

class txtCmdPairsProvide( txtCmdProvide ):
    '''text cmd pairs provide dataset
    Args:
        pathname
    '''
    def __init__(self, 
        pathname, 
        filter=True,
        ):
        super(txtCmdPairsProvide, self).__init__( pathname, filter )
        self.pairs = get_pairs(self.phrases, self.classes, n=10 )

    def __len__(self):
        return len( self.pairs )
    def __getitem__(self, i):
        return self.pairs[i]

    def filter(self, max_length=10):
        self.pairs = filterPairs( self.pairs, max_length )

class txtQuoraProvide( object ):
    '''text quora provide dataset
    Args:
        pathname
    '''
    def __init__(self,  
        pathname, 
        filter=True,
        ):
        self.pathname = pathname      
        #read dataset
        data = pd.read_csv( pathname )
        self.id = data['id']
        self.qid1 = data['qid1']
        self.qid2 = data['qid2']
        self.q1 = data['q1']
        self.q2 = data['q2']
        self.is_duplicate = data['is_duplicate']
        self.q1 = [normalizeString(phrase) for phrase in self.q1 ]
        self.q2 = [normalizeString(phrase) for phrase in self.q2 ]

    def __len__(self):
        return len( self.words )
    def __getitem__(self, i):
        return self.q1[i], self.q2[i]
    def filter(self, max_length=10):
        self.q1 = filterPhrases(self.q1, max_length)
        self.q2 = filterPhrases(self.q2, max_length)

