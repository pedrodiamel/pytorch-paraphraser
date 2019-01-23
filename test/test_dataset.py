import unittest
import os
import sys

sys.path.append('../')

from torchlib.datasets import dataset

class TestDataset( unittest.TestCase ):

#     def test_prepare( self ):
#         pathfile = '~/.datasets/txt/para-nmt-50m-demo/para-nmt-50m-small.txt'
#         pathvocabulary = '~/.datasets/txt/para-nmt-50m-demo/ngram-word-concat-40.pickle'
#         pathfile = os.path.expanduser(pathfile)
#         pathvocabulary = os.path.expanduser(pathvocabulary)

#         print( pathfile )
#         print( pathvocabulary )

#         voc, pairs = dataset.prepare_data( pathfile, pathvocabulary )

#         # print(random.choice(pairs))
#         print(len(pairs))
#         for pair in pairs[:10]:
#             print(pair)

#         self.assertTrue( len(pairs) > 0 )

#     def test_triplet(self):

#         pathname = '~/.datasets/txt'
#         pathfile = 'para-nmt-50m-demo/para-nmt-50m-small.txt'
#         pathvocabulary = 'para-nmt-50m-demo/ngram-word-concat-40.pickle'
#         nbatch = 5
#         batch_size = 10

#         pathname = os.path.expanduser(pathname)
#         pathfile = os.path.expanduser(pathfile)
#         pathvocabulary = os.path.expanduser(pathvocabulary)

#         data = dataset.TxtTripletDataset( 
#             pathname,  
#             pathfile,            
#             pathvocabulary,
#             nbatch=nbatch,
#             batch_size=batch_size,
#         )
        
#         batches = data.getbatch()
#         s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batches

#         # print("s1_variable:", s1)
#         # print("lengths:", s1_max_len)
#         # print("mask:", s1_mask)
#         # print("s2_variable:", s2)
#         # print("lengths:", s2_max_len)
#         # print("mask:", s2_mask)
#         # print("t1_variable:", t1)
#         # print("lengths:", t1_max_len)
#         # print("mask:", t1_mask)

#         self.assertEqual( s1.shape[1], batch_size )
#         self.assertEqual( s2.shape[1], batch_size )
#         self.assertEqual( t1.shape[1], batch_size )


    def test_nmt(self):
        
        pathname = '~/.datasets/txt'
        pathfile = 'para-nmt-50m-demo/para-nmt-50m-small.txt'
        pathvocabulary = 'para-nmt-50m-demo/ngram-word-concat-40.pickle'
        nbatch = 10
        batch_size = 5

        pathname = os.path.expanduser(pathname)
        pathfile = os.path.expanduser(pathfile)
        pathvocabulary = os.path.expanduser(pathvocabulary)

        data = dataset.TxtNMTDataset( 
            pathname,  
            pathfile,
            pathvocabulary,
            nbatch=nbatch,
            batch_size=batch_size,
        )
        
        batches = data.getbatch()
        inp, lengths, output, mask, max_target_len,  = batches

        print("inp: \n", inp)
        print( inp.shape )
        # print("lengths:", lengths)
        print("output: \n", output)
        print( output.shape )
        # print("mask:", mask)
        # print( mask.shape )
        # print("max_target_len:", max_target_len)

        self.assertEqual( inp.shape[1], batch_size )
        self.assertEqual( output.shape[1], batch_size )
                
        all_ref_words = []
        for j in range( batch_size ): 
            tokens_ref = output[:,j]
            decoded_ref_words = [data.voc.index2word[ token.item() ] for token in tokens_ref]
            decoded_ref_words[:] = [x for x in decoded_ref_words if not (x == data.voc.EOS_token or x == data.voc.PAD_token)]
            all_ref_words.append( decoded_ref_words )

        for line in all_ref_words:
            print( line )


if __name__ == '__main__':
    unittest.main()














