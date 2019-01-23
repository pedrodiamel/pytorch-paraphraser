import unittest
import os
import sys
sys.path.append('../')

from torchlib.datasets import embeddings

class TestEmbedding( unittest.TestCase ):

    def test_load_sentence(self):

        pathname = '../rec/data/ngram-word-concat-40.pickle'

        pathname = '~/.datasets/txt'
        pathvocabulary  = 'ngram-word-concat-40.pickle'
        pathname = os.path.expanduser(pathname)
        pathvocabulary = os.path.expanduser(pathvocabulary)

        emb = embeddings.load_sentence_embeddings( os.path.join( pathname, pathvocabulary )  )
        print(emb['n_words'])
        print(emb['index2word'][emb['PAD_token']])
        print(emb['index2word'][emb['SOS_token']])
        print(emb['index2word'][emb['EOS_token']])
        print(emb['index2word'][emb['UNK_token']])


if __name__ == '__main__':
    unittest.main()
