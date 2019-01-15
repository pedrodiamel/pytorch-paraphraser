

import numpy as np
import pickle
from six import iteritems


def load_sentence_embeddings( pathname ):
    '''Read John Wieting sentence embeddings
    https://github.com/vsuthichai/paraphraser
    '''
    with open(pathname , 'rb') as f:
        # [ numpy.ndarray(95283, 300), numpy.ndarray(74664, 300), (trigram_dict, word_dict)]
        x = pickle.load(f, encoding='latin1')            
        word_vocab_size, embedding_size = x[1].shape
        trigram_embeddings, word_embeddings, _ = x
        trigram_to_id, word_to_id = x[2]            
        word_to_id['<START>'] = word_vocab_size
        word_to_id['<END>']   = word_vocab_size + 1                                
        idx_to_word = { idx: word for word, idx in iteritems(word_to_id) }
        word_embeddings = np.vstack((word_embeddings, np.random.randn(2, embedding_size)))
        word_to_count = { word: 1 for word, idx in iteritems(word_to_id) }
        
        return {
            'word2index': word_to_id,
            'index2word': idx_to_word,
            'word2count': word_to_count,
            'embeddings': word_embeddings,
            'n_words':    len(word_to_id),        
            'PAD_token':  word_to_id['★'],        # Used for padding short sentences
            'SOS_token':  word_to_id['<START>'],  # Start-of-sentence token
            'EOS_token':  word_to_id['<END>'],    # End-of-sentence token
            'UNK_token':  word_to_id['UUUNKKK'],  # Unknown
        }

