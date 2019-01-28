
import os 
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocabulary import ( Vocabulary, indexesFromSentence )
from .utils import ( normalizeString )
from .attnet import ( EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder ) 

class NeuralNetNMT( object ):
    r"""Convolutional Neural Net for NMT
    Args:
        no_cuda (bool): system cuda (default is True)
        parallel (bool)
        print_freq (int)
        gpu (int)
    """
    def __init__(self,
        no_cuda=True,
        parallel=False,
        print_freq=10,
        gpu=0
        ):
        
        # cuda
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.parallel = not no_cuda and parallel
        if self.cuda:
            torch.cuda.set_device( gpu )
            
        self.gpu = gpu
        self.encoder=None
        self.decoder=None
        self.embedding=None

    def create(self, 
        arch,
        voc,
        attn_model='dot', 
        hidden_size=300, 
        encoder_n_layers=2, 
        decoder_n_layers=2,
        ):
        """
        Create
        Args:
            arch (string): architecture
            voc (Vocabulary): vocabulary
            attn_model:
            hidden_size:
            encoder_n_layers:
            decoder_n_layers:
        """
        # create models
        self.encoder = None 
        self.decoder = None           
        self.voc = voc
        self.s_arch = arch
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        ## embedded 
        self.embedding = nn.Embedding( voc.n_words, hidden_size )   #74667, voc.n_words
        ## models
        # TODO January 28, 2019: select arch 
        self.encoder   = EncoderRNN( hidden_size, self.embedding, encoder_n_layers )
        self.decoder   = LuongAttnDecoderRNN( attn_model, self.embedding, hidden_size, voc.n_words, decoder_n_layers ) #voc.n_words
        self.search    = GreedySearchDecoder( self.encoder, self.decoder, sos=voc.SOS_token, cuda=self.cuda )
        
        if self.cuda == True:
            self.encoder.cuda( self.gpu )
            self.decoder.cuda( self.gpu )            
        if self.parallel == True and self.cuda == True:
            self.encoder = nn.DataParallel(self.encoder, device_ids=range( torch.cuda.device_count() ))
            self.decoder = nn.DataParallel(self.decoder, device_ids=range( torch.cuda.device_count() ))
            
            

 
    def __call__(self, sentence, max_length):
        ### Format input sentence as a batch
        # normalize
        sentence = normalizeString(sentence)
        # words -> indexes
        indexes_batch = [indexesFromSentence( self.voc, sentence)]    
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)          
        # Decode sentence with searcher
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if self.cuda:
                input_batch = input_batch.cuda() 
                lengths = lengths.cuda()                 
            batch_size = input_batch.shape[1]
            tokens, scores = self.search( input_batch, lengths, max_length, batch_size )    
        # indexes -> words
        UNK = self.voc.index2word[self.voc.UNK_token]     
        decoded_words = [self.voc.index2word.get(token.item(), UNK) for token in tokens] 
        EOS = self.voc.index2word[self.voc.EOS_token]
        PAD = self.voc.index2word[self.voc.PAD_token]
        decoded_words[:] = [x for x in decoded_words if not (x == EOS or x == PAD)]
        return decoded_words, scores

    def load(self, pathnamemodel ):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                self.create( 
                    checkpoint['arch'], 
                    checkpoint['voc'], #checkpoint['voc'], voc <<<----
                    checkpoint['attn_model'], 
                    checkpoint['hidden_size'], 
                    checkpoint['encoder_n_layers'], 
                    checkpoint['decoder_n_layers'] 
                    )                
                self.encoder.load_state_dict( checkpoint['en'] )
                self.decoder.load_state_dict( checkpoint['de'] )
                self.embedding = self.encoder.embedding
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True

            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))   

        return bload

    def __str__(self): 
        return str(
                'Arq: {} \n'
                'Model: \n{} \n{} \n'.format(
                    self.s_arch,
                    self.encoder,
                    self.decoder,
                    )
                )



class Paraphrases( object ):
    r"""Paraphrases
    Args:
    """
    def __init__(self ):
        pass
    
    def run(self, pathconfigurate):

        #load configurate json
        with open(pathconfigurate, "r" ) as f: 
            modelconfig = json.load(f)

        pathmodel      = modelconfig['pathmodel']
#         pathvocabulary = modelconfig['pathvocabulary']
#         namevoc        = modelconfig['namevoc'] 
        namemodel      = modelconfig['namemodel'] 
        max_length     = modelconfig['max_length']
        parallel       = modelconfig['parallel']
        no_cuda        = modelconfig['no_cuda']
        gpu            = modelconfig['gpu']    
        
#         # load vocabulary
#         print('>> Load vocabulary ...')
#         pathvocabulary = os.path.join( os.path.expanduser( pathvocabulary ), namevoc )   
#         voc = Vocabulary()
#         voc.load_embeddings( pathvocabulary, type='emb' ) 
#         print(">> Counted words:")
#         print(voc.n_words)

        # load model 
        print('>> Load model ...')
        pathmodel = os.path.join( os.path.expanduser( pathmodel ), namemodel )        
        network = NeuralNetNMT(
            no_cuda=no_cuda,
            parallel=parallel,
            gpu=gpu
            )

        if network.load( pathmodel ) is not True:
            raise ValueError('Error: model not load ...')
        print( network )

        self.max_length = max_length
        self.network = network
        


    def __call__(self, sentence):
        # evaluate        
        decoded_words, score = self.network( sentence, self.max_length  )    
        return decoded_words, score

