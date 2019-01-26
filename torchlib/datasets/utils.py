
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


def filterPair(p, max_length=10 ):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length 


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

def get_pairs( word, types, n=5 ):  
    '''get pairs 
    Args:
        word: list of words 
        types: list of words types 
        n: represent the number of combination for each class 
    '''  
    pairs = [] 
    C,F = np.unique( types, return_counts=True )
    for c,f in zip(C,F):         
        # a, b   
        a = np.array( np.random.choice( np.where(types==c)[0], min(f,n), replace=False ))
        b = np.array( np.random.choice( np.where(types==y)[0], min(f,n), replace=False ))
        #while np.any((a-b)==0): #aligning check
        while np.sum((a-b) == 0 )/b.shape[0] > 0.1: #aligning check
            random.shuffle(b) 
        pairs += zip(word[a],word[b])
        #print(c, a,b)
    random.shuffle(pairs)
    return pairs

def read_pairs( pathname ):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(pathname, encoding='utf-8').\
        read().strip().split('\n')    
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]    
    return pairs

def read_word2pairs( pathname ):     
    return get_pairs( *read_words(pathname) )


def read_words( pathname ):
    '''read words dataset
    Format: type; word
    '''
    data = pd.read_csv( pathname )
    words, types = data['word'], data['type']
    return words, types

