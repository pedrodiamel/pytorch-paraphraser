

import sys
sys.path.append('../')
import pytest

from torchlib.datasets import utils


def test_read():
    pathname = '../rec/data/para-nmt-50m-small.txt'
    pairs = utils.read_paraphraser( pathname )
    print( len(pairs) )
    print( pairs[1] )

def test_normalize_string():    
    s = 'This Is a tEst ... '
    s_norm = utils.normalizeString(s)
    print( '>> {}'.format(s) )
    print( '<< {}'.format(s_norm) )

# test_read()
test_normalize_string()