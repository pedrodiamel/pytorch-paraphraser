

import os
import random 
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

from torchlib.datasets.dataset  import TxtPairDataset
from torchlib.tripletneuralnet import NeuralNetTripletNLP


def random_query( SS1, SS2, SENC1, SENC2, nq=1, maxq=5 ):

    for i in range( nq ):   

        #random quary
        k = np.random.randint( len(SS1) )
        ss1 = SS1[ k ]
        s1_enc = SENC1[k,:]   

        #dist = cosine_similarity( s1_enc, SENC2[0,:] )
        dist = np.array([ cosine_similarity( s1_enc.reshape(1, -1), s.reshape(1, -1) )[0,0] for s in SENC2 ])
        indx = np.argsort( dist )
        
        # print(indx)
        # print(indx[::-1]  )

        indx = indx[::-1]         

        print( '>> Query: ', ss1, )
        print( '>> Expected: ', SS2[k]  )
        print( '>> Result: ')
        for j in range( maxq ):
            print( '<< [{}]: {}'.format( j, SS2[ indx[j] ] ) )
        print('')


def main():

    pathname       = '~/.datasets/txt'
    namedataset    = 'paranmt' #cmds, paranmt,
    pathdata       = 'commandpairsextsmall.txt' #dbcommand.csv; commandpairsext.txt; para-nmt-50m/para-nmt-50m.txt; para-nmt-50m-demo/para-nmt-50m-small.txt 
    pathvocabulary = 'para-nmt-50m-demo/ngram-word-concat-40.pickle'
    pathmodel      = 'out/netruns/nlp_nmt_maskll_adam_paranmt_004/models/model_best.pth.tar'
    nbatch=0
    batch_size=None
    max_length=10

    project='./out/netruns'
    name='nlp_encoder_rnn_avg_tripletloss_adam_paranmt_005'
    no_cuda=False
    seed=0
    gpu=0
    parallel=False
    pathmodel=os.path.join( project, name, 'models/model_best.pth.tar' )

    # datasets
    # training dataset
    dataset = TxtPairDataset(
        pathname=pathname,
        namedataset=namedataset,
        filedataset=pathdata,
        filevocabulary=pathvocabulary,
        nbatch=nbatch,
        batch_size=batch_size,
        max_length=max_length,
    )
        

    print( len(dataset) )

    #neural net 
    #get model
    network = NeuralNetTripletNLP(
        patchproject=project,
        nameproject=name,
        no_cuda=no_cuda,
        parallel=parallel,
        seed=seed,
        gpu=gpu
        )

    if network.load( pathmodel ) is not True:
        raise ValueError('Error: model not load ...')
    
    print( 'load neuralnet ... ' )
    print( network )

    #pair_encoder = network.predict( dataset )
    #print( pair_encoder.shape )

    #representation 
    network.net.eval()
    network.encoder.eval()

    SS1, SS2, SENC1, SENC2 = [],[],[],[]
    with torch.no_grad():
        for i in range( len(dataset) ):
            sample = dataset[i]
            ss1, s1, s1_mask, s1_len, ss2, s2, s2_mask, s2_len = sample
        
            if not no_cuda:
               s1 = s1.cuda(); s1_mask = s1_mask.cuda()
               s2 = s2.cuda(); s2_mask = s2_mask.cuda()           

            s1_enc = network.encoder( s1, s1_mask )
            s2_enc = network.encoder( s2, s2_mask )
            SS1.append(ss1)
            SS2.append(ss2)
            SENC1.append(s1_enc)
            SENC2.append(s2_enc)
            
            if i > 100:
                break

    SENC1 = np.concatenate( SENC1, axis=0 )
    SENC2 = np.concatenate( SENC2, axis=0 )

    print( SENC1.shape )
    print( SENC2.shape )

    random_query( SS1, SS2, SENC1, SENC2 , nq=2 )
    

if __name__ == '__main__':
    main()












# import random
# import torch
# from torchlib.datasets.vocabulary import ( indexesFromSentence, zeroPadding, binaryMatrix)


# def evaluate(encmodel, voc, sentence, max_length=10):    
#     # words -> indexes
#     indexes_batch = [indexesFromSentence(voc, sentence)]
#     max_target_len = max([len(indexes) for indexes in indexes_batch])
#     padList = zeroPadding(indexes_batch, voc.EOS_token)
#     mask = binaryMatrix(padList, voc.EOS_token)
#     mask = torch.ByteTensor(mask)
#     input_batch = torch.LongTensor(padList) #.transpose(0, 1)        
#     input_batch = input_batch.to(device)
#     mask = mask.to(device)
#     input_enc = encmodel( input_batch,  mask )      
#     return input_enc


# def evaluateRandomly(net, voc, pair, n=5):
#     for i in range(n):
#         pair = random.choice(pairs)
#         input_sentence_s1 = normalizeString( pair[0] )
#         input_sentence_s2 = normalizeString( pair[1] )
#         pair = random.choice(pairs)
#         input_sentence_t1 = normalizeString( pair[0] )
                
#         print('>', input_sentence_s1)
#         print('>', input_sentence_s2)
#         print('>', input_sentence_t1)        
        
#         input_sentence_s1_enc = net(encmodel, voc, input_sentence_s1 )
#         input_sentence_s2_enc = net(encmodel, voc, input_sentence_s2 )
#         input_sentence_t1_enc = net(encmodel, voc, input_sentence_t1 )
        
#         eps=1e-6
#         dist_s1s2 = F.cosine_similarity(input_sentence_s1_enc, input_sentence_s2_enc, dim=1, eps=eps)
#         dist_s1t1 = F.cosine_similarity(input_sentence_s1_enc, input_sentence_t1_enc, dim=1, eps=eps)
                
#         output = '{:.3f}|{:.3f}'.format(dist_s1s2.item(), dist_s1t1.item())
        
#         print('<', output)
#         print('')

        
# # ['of course you did .', 'of course it is .']
# # [' why not ?', ' why not ?']
# input_sentence = 'of course you did .'
# input_sentence = normalizeString(input_sentence)
# print(input_sentence)

# input_enc = evaluate(encoder, dictionary, input_sentence)
# print(input_enc.shape)

# evaluateRandomly(encoder, dictionary, pairs)