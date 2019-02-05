

import os
import time
import random 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models as netmodels
from . import graphic as gph
from . import netlearningrate
from . import utils
from . import loss as netloss
from . import logger

from .neuralnet import  NeuralNetAbstractNLP


class NeuralNetTripletNLP(NeuralNetAbstractNLP):
    r"""Convolutional Neural Net for Triplet NLP
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
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        super(NeuralNetTripletNLP, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.encoder=None

 
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
        cfg_scheduler= { 'step_size':30, 'gamma':0.1  }
                    
        super(NeuralNetTripletNLP, self).create( 
            arch, 
            voc,
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            pretrained, 
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
        )
        
        self.accuracy = netloss.Accuracy( )

        # Set the graphic visualization
        self.logger_train = logger.Logger( 'Trn', ['loss'], ['acc'], self.plotter  )
        self.logger_val   = logger.Logger( 'Val', ['loss'], ['acc'], self.plotter  )
    
    def training(self, dataset, epoch=0):

        self.logger_train.reset()
        data_time = logger.AverageMeter()
        batch_time = logger.AverageMeter()

        # switch to evaluate mode
        self.net.train()
        end = time.time()
        for i, batch in enumerate( dataset.getbatchs() ):
            
            # measure data loading time
            data_time.update(time.time() - end)
            # get data 
            s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batch
            batch_size = s1.shape[0]

            if self.cuda:
                s1 = s1.cuda(); s1_mask = s1_mask.cuda()
                s2 = s2.cuda(); s2_mask = s2_mask.cuda()
                t1 = t1.cuda(); t1_mask = t1_mask.cuda()
            
            # fit (forward)
            emb_s1, emb_s2, emb_t1 = self.net( s1, s1_mask, s2, s2_mask, t1, t1_mask )

            # measure accuracy and record loss
            loss = self.criterion( emb_s1, emb_s2, emb_t1  )            
            pred = self.accuracy( emb_s1, emb_s2, emb_t1  )
              
            # optimizer
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients: gradients are modified in place
            #_ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
            self.optimizer.step()
                      
            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                {'acc': pred.data[0] },  
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
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate( dataset.getbatchs() ):
                
                # get data (image, label)
                s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len, t1, t1_mask, t1_max_len = batch
                batch_size = s1.shape[0]

                if self.cuda:
                    s1 = s1.cuda(); s1_mask = s1_mask.cuda()
                    s2 = s2.cuda(); s2_mask = s2_mask.cuda()
                    t1 = t1.cuda(); t1_mask = t1_mask.cuda()
                
                # fit (forward)
                emb_s1, emb_s2, emb_t1 = self.net( s1, s1_mask, s2, s2_mask, t1, t1_mask )

                # measure accuracy and record loss
                loss = self.criterion( emb_s1, emb_s2, emb_t1  )            
                pred = self.accuracy( emb_s1, emb_s2, emb_t1  ) 

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                {'loss': loss.data[0] },
                {'acc': pred.data[0] },
                batch_size,
                )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(dataset), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['acc'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(dataset), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )
                      
        return acc
 
    def test(self, dataset):
        k=0
        P=[]        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate( tqdm( dataset.getbatchs() ) ):
                # get data
                s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len = batch                             
                if self.cuda:
                    s1 = s1.cuda(); s1_mask = s1_mask.cuda()
                    s2 = s2.cuda(); s2_mask = s2_mask.cuda()
                # fit (forward)
                s1_enc = self.encoder( s1, s1_mask )
                s2_enc = self.encoder( s2, s2_mask )
                p_enc = torch.stack( (s1_enc, s2_enc), dim=2 )
                P.append( p_enc )
        P = torch.cat( P, dim=0 )
        return P

    def predict(self, dataset):        
        P_enc=[]; k=0      
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, batch in enumerate( tqdm( dataset.getbatchs() ) ):
                # get data
                s1, s1_mask, s1_max_len, s2, s2_mask, s2_max_len = batch                             
                if self.cuda:
                    s1 = s1.cuda(); s1_mask = s1_mask.cuda()
                    s2 = s2.cuda(); s2_mask = s2_mask.cuda()
                # fit (forward)
                s1_enc = self.encoder( s1, s1_mask )
                s2_enc = self.encoder( s2, s2_mask )
                p_enc = torch.stack( (s1_enc, s2_enc), dim=2 )
                P_enc.append( p_enc )
        P_enc = torch.cat( P_enc, dim=0 )
        return P_enc

    def __call__(self, x, x_mask):       
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            if self.cuda:
                x = x.cuda()
                x_mask = x_mask.cuda() 
            x_enc = self.encoder( x, x_mask )
        return x_enc
    
    def _create_model(self, arch, voc, pretrained):
        """
        Create model
            @arch (string): select architecture
            @embedding:
            @pretrained (bool)
        """    

        self.net = None
        self.encoder = None   
        self.voc = voc         

        hidden_size = 300
        self.embedding = nn.Embedding( voc.n_words, hidden_size ) 
        #if embedding is None:
        #   embedding = nn.Embedding( voc.n_words, hidden_size ) 
        #elif isinstance(data, torch.Tensor):
        #   embedding = embedding
        #else:   
        #embedding = nn.Embedding.from_pretrained( torch.from_numpy( voc.embeddings ).float(),  freeze=False )
        
        kw = {'embedding': self.embedding, 'pretrained': pretrained}
        self.encoder = netmodels.__dict__[arch](**kw)        
        self.net = netmodels.Tripletnet( self.encoder )        
        self.s_arch = arch

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids=range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'tripletloss':
            self.criterion = netloss.TripletCosineLoss().cuda()
        else:
            assert(False)
        self.s_loss = loss
