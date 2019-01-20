
import sys
sys.path.append('../')
import pytest

from torchlib.datasets import dataset


def test_prepare_data():
    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    voc, pairs = dataset.prepare_data( pathname, pathvocabulary )

    # print(random.choice(pairs))
    print(len(pairs))
    for pair in pairs[:10]:
        print(pair)


def test_triplet_dataset():

    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtTripletDataset( 
        pathname,  
        pathvocabulary,
        nbatch=10,
        batch_size=5,
    )
    
    batches = data.getbatch()
    s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batches

    print("s1_variable:", s1)
    print("lengths:", s1_max_len)
    print("mask:", s1_mask)
    print("s2_variable:", s2)
    print("lengths:", s2_max_len)
    print("mask:", s2_mask)
    print("t1_variable:", t1)
    print("lengths:", t1_max_len)
    print("mask:", t1_mask)


def test_pairs_dataset():
    
    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtPairDataset( 
        pathname,  
        pathvocabulary,
        nbatch=10,
        batch_size=5,
    )
    
    batches = data.getbatch()
    s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len = batches

    print("s1_variable:", s1)
    print("lengths:", s1_max_len)
    print("mask:", s1_mask)
    print("s2_variable:", s2)
    print("lengths:", s2_max_len)
    print("mask:", s2_mask)


def test_nmt_dataset():
    
    pathname = '../rec/data/para-nmt-50m-small.txt'
    pathvocabulary = '../rec/data/ngram-word-concat-40.pickle'
    data = dataset.TxtNMTDataset( 
        pathname,  
        pathvocabulary,
        nbatch=10,
        batch_size=5,
    )
    
    batches = data.getbatch()
    inp, lengths, output, mask, max_target_len,  = batches

    print("inp:", inp)
    print( inp.shape )
    print("lengths:", lengths)
    print("output:", output)
    print( output.shape )
    print("mask:", mask)
    print( mask.shape )
    print("max_target_len:", max_target_len)




# test_prepare_data()
# test_triplet_dataset()
# test_pairs_dataset()
test_nmt_dataset()