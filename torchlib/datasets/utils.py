
import os
from io import open
import unicodedata
import string
import re
import random



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


def read_paraphraser( pathname ):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(pathname, encoding='utf-8').\
        read().strip().split('\n')    
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]    
    return pairs
