

import os
import time
import random 
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import vocabulary as voc
from .models import attnet 

from . import models as netmodels
from . import graphic as gph
from . import netlearningrate
from . import utils
from . import loss as netloss
from . import logger

from .neuralnet import  NeuralNetAbstractNLP

class NeuralNetNMT(NeuralNetAbstractNLP):
    r"""Convolutional Neural Net for NMT
    Args:
        patchproject (str): path project
        nameproject (str):  name project
        no_cuda (bool): system cuda (default is True)
        parallel (bool)
        seed (int)
        print_freq (int)
        gpu (int)
    """

    def __init__(self,
        patchproject='',
        nameproject='',
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        super(NeuralNetNMT, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )

        # initialization
        self.encoder_lr = 0.0001
        self.decoder_lr = 0.0001
 
        self.net = None
        self.encoder=None
        self.decoder=None
        self.embedding=None
        self.criterion = None

        self.encoder_optimizer = None
        self.decoder_optimizer = None       
        self.encoder_lrscheduler = None
        self.decoder_lrscheduler = None

  
    def create(self, 
        arch, 
        voc,
        loss, 
        lr,          
        optimizer, 
        lrsch,  
        momentum=0.9,
        weight_decay=5e-4,        
        pretrained=False,
        attn_model='dot', 
        hidden_size=300, 
        encoder_n_layers=2, 
        decoder_n_layers=2,
        clip=50.0,
        dropout=0.1,
        teacher_forcing_ratio=1.0,
        ):
        """
        Create
        Args:
            arch (string): architecture
            voc:
            loss (string):
            lr (float): learning rate
            momentum,
            optimizer (string) : 
            lrsch (string): scheduler learning rate
            pretrained (bool)
        """


        cfg_opt= { 'momentum':0.9, 'weight_decay':5e-4 } 
        cfg_scheduler= { 'step_size':100, 'gamma':0.1  }
        cfg_model = {
            'attn_model': attn_model, 
            'hidden_size': hidden_size, 
            'encoder_n_layers': encoder_n_layers, 
            'decoder_n_layers': decoder_n_layers,
            'dropout': dropout,
        }
                    
        super(NeuralNetNMT, self).create( 
            arch,
            voc,
            loss,
            lr,
            optimizer,
            lrsch,
            pretrained,
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
            cfg_model=cfg_model,
        )

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip = clip
        self.accuracy = netloss.Bleu()

        # Set the graphic visualization
        self.logger_train = logger.Logger( 'Trn', ['loss'], ['bleu'], self.plotter  )
        self.logger_val   = logger.Logger( 'Val', ['loss'], ['bleu'], self.plotter  )
        

    def _forward(self, 
        input_variable, lengths, target_variable, 
        mask, max_target_len, 
        batch_size, 
        teacher_forcing_ratio, 
        clip
        ):

        # Initialize variables
        loss = 0
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder( input_variable, lengths )

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.voc.SOS_token for _ in range( batch_size )]])
        if self.cuda:
            decoder_input = decoder_input.cuda()

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal =  self.criterion(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.cuda() if self.cuda else decoder_input

                # Calculate and accumulate loss
                mask_loss, nTotal = self.criterion(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                n_totals += nTotal

        return loss, n_totals
    
    def training(self, dataset, epoch=0):

        self.logger_train.reset()
        data_time = logger.AverageMeter()
        batch_time = logger.AverageMeter()

        # switch to evaluate mode
        self.encoder.train()
        self.decoder.train()

        end = time.time()
        for i, batch in enumerate( dataset.getbatchs() ):
            
            # measure data loading time
            data_time.update(time.time() - end)
            # get data 
            inp, lengths, output, mask, max_target_len = batch
            batch_size = inp.shape[1]             

            if self.cuda:
                inp = inp.cuda()
                lengths = lengths.cuda()
                output = output.cuda()
                mask = mask.cuda()
                
            # fit (forward)
            loss, n_totals = self._forward( inp, lengths, output, 
                mask, max_target_len, 
                batch_size, 
                self.teacher_forcing_ratio, 
                self.clip
                )
            #loss = loss/n_totals
              
            # optimizer
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
            loss.backward()
            loss /=n_totals

            # Clip gradients: gradients are modified in place
            _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

            # Adjust model weights
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                {'bleu': 0 },  
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(dataset), i, len(dataset), batch_time,   )

    def evaluate(self, dataset, epoch=0):
        
        self.logger_val.reset()
        batch_time = logger.AverageMeter()        

        # switch to evaluate mode
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate( dataset.getbatchs() ):
                
                # get data (image, label)
                inp, lengths, output, mask, max_target_len = batch
                batch_size = inp.shape[1]

                if self.cuda:
                    inp = inp.cuda()
                    lengths = lengths.cuda()
                    output = output.cuda()
                    mask = mask.cuda()
                                    
                # fit (forward)
                loss, n_totals = self._forward( 
                    inp, lengths, output, 
                    mask, max_target_len, 
                    batch_size, 
                    self.teacher_forcing_ratio, 
                    self.clip
                    )
                loss = loss/n_totals
    
                # metrics accuracy                      
                tokens_batch, scores_batch = self.search( inp , lengths, max_target_len, batch_size )
                
                all_hyp_words = []
                all_ref_words = []
                EOS = self.voc.index2word[self.voc.EOS_token]
                PAD = self.voc.index2word[self.voc.PAD_token]
                UNK = self.voc.index2word[self.voc.UNK_token]                
                for j in range( batch_size ): 
                    tokens_hyp = tokens_batch[:,j]
                    tokens_ref = output[:,j]
                    
                    #decoded_hyp_words = [self.voc.index2word[ token.item() ] for token in tokens_hyp]
                    decoded_hyp_words = [ self.voc.index2word.get( token.item(), UNK ) for token in tokens_hyp ] 
                    decoded_hyp_words[:] = [x for x in decoded_hyp_words if not (x == EOS or x == PAD)]
                    all_hyp_words.append( decoded_hyp_words )
                    
                    #decoded_ref_words = [ self.voc.index2word[ token.item() ] for token in tokens_ref]
                    decoded_ref_words = [self.voc.index2word.get( token.item(), UNK ) for token in tokens_ref ] 
                    decoded_ref_words[:] = [x for x in decoded_ref_words if not (x == EOS or x == PAD)]                    
                    all_ref_words.append( decoded_ref_words )
                
                
                all_ref_words = [ [ref] for ref in all_ref_words ]
                blue = corpus_bleu(all_ref_words, all_hyp_words) 

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                {'loss': loss },
                {'bleu': blue },
                batch_size,
                )

                if i % self.print_freq == 0: 
                    
                    print()
                    print('>>', all_ref_words[0] )
                    print('>>', all_hyp_words[0] )
                    print()
                    
                    self.logger_val.logger(
                        epoch, epoch, i,len( dataset ), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        bleu = self.logger_val.info['metrics']['bleu'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(dataset), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )
                      
        return bleu
 
    def test(self, dataset):
        
        bleus = []

        # switch to evaluate mode
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            end = time.time()
            for i, batch in  enumerate( tqdm(dataset.getbatchs() ) ):
                # get data
                inp, lengths, output, mask, max_target_len = batch                            
                batch_size = inp.shape[1]

                if self.cuda:
                    inp = inp.cuda()
                    lengths = lengths.cuda()
                    output = output.cuda()
                    mask = mask.cuda()
                    
                # fit (forward)
                tokens_batch, scores_batch = self.search( inp , lengths, max_target_len, batch_size )

                all_hyp_words = []
                all_ref_words = []
                EOS = self.voc.index2word[self.voc.EOS_token]
                PAD = self.voc.index2word[self.voc.PAD_token]
                UNK = self.voc.index2word[self.voc.UNK_token]                
                for j in range( batch_size ): 
                    tokens_hyp = tokens_batch[:,j]
                    tokens_ref = output[:,j]
                    
                    #decoded_hyp_words = [self.voc.index2word[ token.item() ] for token in tokens_hyp]
                    decoded_hyp_words = [ self.voc.index2word.get( token.item(), UNK ) for token in tokens_hyp ] 
                    decoded_hyp_words[:] = [x for x in decoded_hyp_words if not (x == EOS or x == PAD)]
                    all_hyp_words.append( decoded_hyp_words )
                    
                    #decoded_ref_words = [ self.voc.index2word[ token.item() ] for token in tokens_ref]
                    decoded_ref_words = [self.voc.index2word.get( token.item(), UNK ) for token in tokens_ref ] 
                    decoded_ref_words[:] = [x for x in decoded_ref_words if not (x == EOS or x == PAD)]                    
                    all_ref_words.append( decoded_ref_words )
                
                all_ref_words = [ [ref] for ref in all_ref_words ]
                bleu = corpus_bleu(all_ref_words, all_hyp_words) 
                bleus.append( bleu )
        
        bleus = np.stack( bleus, axis=-1 ).mean()
        return  bleus


    def __call__(self, x, lengths, max_target_len ):       
        # switch to evaluate mode
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if self.cuda:
                x = x.cuda() 
                lengths = lengths.cuda()                 
            batch_size = x.shape[1]
            tokens_batch, scores_batch = self.search( x, lengths, max_target_len, batch_size )
        return tokens_batch, scores_batch


    def _create_model(self, arch, voc, pretrained, attn_model, hidden_size, encoder_n_layers, decoder_n_layers, dropout=0.1):
        """
        Create model
            arch (string): select architecture
            voc:
            pretrained (bool)
            attn_model 
            hidden_size 
            encoder_n_layers 
            decoder_n_layers 
            dropout
        """    

        self.net = None
        self.encoder = None 
        self.decoder = None           
        self.voc = voc
        self.s_arch = arch
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers

        #if embedding is None:
        self.embedding = nn.Embedding( voc.n_words, hidden_size )         
        #elif isinstance(data, torch.Tensor):
        #   self.embedding = embedding
        #else:   
        #self.embedding = nn.Embedding.from_pretrained( torch.from_numpy( voc.embeddings ).float(),  freeze=False )       
        
        #kw = {'embedding': self.embedding, 'pretrained': pretrained}
        #self.encoder = netmodels.__dict__[arch](**kw)        
        
        self.encoder   = attnet.EncoderRNN( hidden_size, self.embedding, encoder_n_layers, dropout )
        self.decoder   = attnet.LuongAttnDecoderRNN( attn_model, self.embedding, hidden_size, voc.n_words, decoder_n_layers, dropout )
        self.search    = attnet.GreedySearchDecoder( self.encoder, self.decoder, sos=voc.SOS_token, cuda=self.cuda )
                
        if self.cuda == True:
            self.encoder.cuda()
            self.decoder.cuda()
        if self.parallel == True and self.cuda == True:
            self.encoder = nn.DataParallel(self.encoder, device_ids=range( torch.cuda.device_count() ))
            self.decoder = nn.DataParallel(self.decoder, device_ids=range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'maskll':
            self.criterion = netloss.MaskNLLLoss().cuda()
        else:
            assert(False)
        self.s_loss = loss

    def _create_optimizer(self, optimizer='adam', lr=0.0001, **kwargs ):
        """
        Create optimizer
        Args:
            @optimizer (string): select optimizer function
            @lr (float): learning rate
            @momentum (float): momentum
        """
        
        self.optimizer = None
        learning_rate=0.5

        # create optimizer
        if optimizer == 'adam':
            self.encoder_optimizer = torch.optim.Adam( self.encoder.parameters(), lr=lr, amsgrad=True ) 
            self.decoder_optimizer = torch.optim.Adam( self.decoder.parameters(), lr=lr*learning_rate, amsgrad=True )  

        elif optimizer == 'sgd':
            self.encoder_optimizer = torch.optim.SGD( self.encoder.parameters(), lr=lr, **kwargs ) 
            self.decoder_optimizer = torch.optim.SGD( self.decoder.parameters(), lr=lr, **kwargs ) 

        elif optimizer == 'rprop':
            self.encoder_optimizer = torch.optim.Rprop( self.encoder.parameters(), lr=lr) 
            self.decoder_optimizer = torch.optim.Rprop( self.decoder.parameters(), lr=lr) 

        elif optimizer == 'rmsprop':
            self.encoder_optimizer = torch.optim.RMSprop( self.encoder.parameters(), lr=lr)           
            self.decoder_optimizer = torch.optim.RMSprop( self.decoder.parameters(), lr=lr)      

        else:
            assert(False)

        self.encoder_lr = lr
        self.decoder_lr = lr*learning_rate
        self.s_optimizer = optimizer

    def _create_scheduler_lr(self, lrsch, **kwargs ):
        
        #MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        #ExponentialLR
        #CosineAnnealingLR

        self.lrscheduler = None

        if lrsch == 'fixed':
            pass           
        elif lrsch == 'step':
            self.encoder_lrscheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, **kwargs ) # step_size=3, gamma=0.1 
            self.decoder_lrscheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, **kwargs ) # step_size=3, gamma=0.1 
        elif lrsch == 'cyclic': 
            self.encoder_lrscheduler = netlearningrate.CyclicLR(self.encoder_optimizer, **kwargs)
            self.decoder_lrscheduler = netlearningrate.CyclicLR(self.decoder_optimizer, **kwargs)            
        elif lrsch == 'exp':
            self.encoder_lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.encoder_optimizer, **kwargs ) # gamma=0.99 
            self.decoder_lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.decoder_optimizer, **kwargs ) # gamma=0.99 
        elif lrsch == 'plateau':
            self.encoder_lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, **kwargs ) # 'min', patience=10
            self.decoder_lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, **kwargs ) # 'min', patience=10
        else:
            assert(False)
        self.s_lerning_rate_sch = lrsch

    def adjust_learning_rate(self, epoch ):
        """
        Update learning rate
        """        
        # update
        if self.s_lerning_rate_sch == 'fixed': 
            lr_encoder = self.encoder_lr
            lr_decoder = self.encoder_lr
        elif self.s_lerning_rate_sch == 'plateau':
            self.encoder_lrscheduler.step( self.vallosses )
            for param_group in self.encoder_optimizer.param_groups:
                lr_encoder = float(param_group['lr'])
                break            
            self.decoder_lrscheduler.step( self.vallosses )
            for param_group in self.decoder_optimizer.param_groups:
                lr_decoder = float(param_group['lr'])
                break      
        else:                    
            self.encoder_lrscheduler.step() 
            self.decoder_lrscheduler.step() 
            lr_encoder = self.encoder_lrscheduler.get_lr()[0]  
            lr_decoder = self.decoder_lrscheduler.get_lr()[0]       

        # draw
        self.plotter.plot('lr', 'enc learning rate', epoch, lr_encoder )
        self.plotter.plot('lr', 'dec learning rate', epoch, lr_decoder )
 
    def resume(self, resume):
        """
        Resume: optionally resume from a checkpoint
        """ 
        #net = self.net.module if self.parallel else self.net
        encoder = self.encoder.module if self.parallel else self.encoder
        decoder = self.decoder.module if self.parallel else self.decoder
        start_epoch = 0
        prec = 0
        if resume:
            if os.path.isfile(resume):
                
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                
                start_epoch = checkpoint['epoch']
                prec = checkpoint['prec']
                encoder.load_state_dict(checkpoint['en'])
                decoder.load_state_dict(checkpoint['de'])
                
                self.encoder_optimizer.load_state_dict(checkpoint['en_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['de_optimizer'])                    
                self.embedding = encoder.embedding

                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
        self.start_epoch = start_epoch
        return start_epoch, prec

    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        #net = self.net.module if self.parallel else self.net
        encoder = self.encoder.module if self.parallel else self.encoder
        decoder = self.decoder.module if self.parallel else self.decoder
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'attn_model': self.attn_model,
                'hidden_size': self.hidden_size,
                'encoder_n_layers': self.encoder_n_layers,
                'decoder_n_layers': self.decoder_n_layers,
                'prec': prec,                   
                'en': self.encoder.state_dict(),
                'de': self.decoder.state_dict(),
                'en_optimizer' : self.encoder_optimizer.state_dict(),
                'de_optimizer' : self.decoder_optimizer.state_dict(),

            }, 
            is_best,
            self.pathmodels,
            filename,
            )
   
    def load(self, pathnamemodel, voc):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                self._create_model( 
                    checkpoint['arch'], 
                    voc, 
                    False, 
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
                'Name: {} \n'
                'arq: {} \n'
                'loss: {} \n'
                'optimizer: {} \n'
                'lr: {} \n'
                'Model: \n{} \n{} \n'.format(
                    self.nameproject,
                    self.s_arch,
                    self.s_loss,
                    self.s_optimizer,
                    self.lr,
                    self.encoder,
                    self.decoder,
                    )
                )