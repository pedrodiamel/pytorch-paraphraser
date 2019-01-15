

import os
import time
import random 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import vocabulary as voc

from . import models as netmodels
from . import graphic as gph
from . import netlearningrate
from . import utils
from . import loss as netloss
from . import logger


#----------------------------------------------------------------------------------------------
# Neural Net NLP Class

class NeuralNetAbstractNLP(object):
    """
    Abstract NLP Neural Net 
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
        """
        Initialization
        Args:
            @patchproject (str): path project
            @nameproject (str):  name project
            @no_cuda (bool): system cuda (default is True)
            @parallel (bool)
            @seed (int)
            @print_freq (int)
            @gpu (int)
        """

        # cuda
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.parallel = not no_cuda and parallel
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.set_device( gpu )
            torch.cuda.manual_seed(seed)

        # set project directory
        self.nameproject = nameproject
        self.pathproject = os.path.join(patchproject, nameproject)

        # Set the graphic visualization
        self.plotter = None
        self.plotter = gph.VisdomLinePlotter(env_name=self.nameproject)

        
        # initialization var
        self.print_freq = print_freq
        self.lr = 0.0001
        self.start_epoch = 0        

        self.s_arch = ''
        self.s_optimizer = ''
        self.s_lerning_rate_sch = ''
        self.s_loss = ''

        self.net = None
        self.encoder=None
        self.embedding=None
        self.criterion = None
        self.optimizer = None
        self.lrscheduler = None
        self.vallosses = 0

    def create(self, 
        arch,
        embedded, 
        loss, 
        lr, 
        optimizer, 
        lrsch, 
        pretrained=False,
        cfg_model=None,
        cfg_loss=None,
        cfg_opt=None,
        cfg_scheduler=None,
        ):
        """
        Create 
        Args:
            @arch (string): architecture
            @loss (string):
            @lr (float): learning rate
            @optimizer (string) : 
            @lrsch (string): scheduler learning rate
            @pretrained (bool)
        """
                
        self.s_arch = arch
        self.s_optimizer = optimizer
        self.s_lerning_rate_sch = lrsch
        self.s_loss = loss

        if cfg_model is None:       #defaul configurate
            cfg_model = {}
        if cfg_loss is None:        #defaul configurate
            cfg_loss = {}
        if cfg_opt is None:         #defaul configurate
            cfg_opt = {}
        if cfg_scheduler is None:   #defaul configurate
            cfg_scheduler = {}

        # create project directory
        self.pathmodels = os.path.join(self.pathproject, 'models')
        if not os.path.exists(self.pathproject):
            os.makedirs(self.pathproject)
        if not os.path.exists(self.pathmodels):
            os.makedirs(self.pathmodels)

        # Set the graphic visualization
        #self.plotter = gph.VisdomLinePlotter(env_name=self.nameproject)
                        
        self._create_model( arch, embedded, pretrained, **cfg_model )
        self._create_loss( loss, **cfg_loss )
        self._create_optimizer( optimizer, lr, **cfg_opt )
        self._create_scheduler_lr( lrsch, **cfg_scheduler )

        

    def training(self, data_loader, epoch=0):
        pass

    def evaluate(self, data_loader, epoch=0):
        pass

    def test(self, data_loader):
        pass

    def inference(self, image):        
        pass

    def representation(self, data_loader):
        pass
    
    def fit( self, train_loader, val_loader, epochs=100, snapshot=10 ):

        best_prec = 0
        print('\nEpoch: {}/{}(0%)'.format(self.start_epoch, epochs))
        print('-' * 25)

        self.evaluate(val_loader, epoch=self.start_epoch)        
        for epoch in range(self.start_epoch, epochs):       

            try:
                
                self._to_beging_epoch(epoch, epochs, train_loader, val_loader)

                self.adjust_learning_rate(epoch)     
                self.training(train_loader, epoch)

                print('\nEpoch: {}/{} ({}%)'.format(epoch,epochs, int((float(epoch)/epochs)*100) ) )
                print('-' * 25)
                
                prec = self.evaluate(val_loader, epoch+1 )            

                # remember best prec@1 and save checkpoint
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
                if epoch % snapshot == 0 or is_best or epoch==(epochs-1) :
                    self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))

                self._to_end_epoch(epoch, epochs, train_loader, val_loader)

            except KeyboardInterrupt:
                
                print('Ctrl+C, saving snapshot')
                is_best = False
                best_prec = 0
                self.save(epoch, best_prec, is_best, 'chk{:06d}.pth.tar'.format(epoch))
                return

    def _to_beging_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass

    def _to_end_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass

    def _create_model(self, arch, embedded, pretrained):
        """
        Create model
        Args:
            @arch (string): select architecture
            @embedded
            @pretrained (bool)
        """    
        pass

    def _create_loss(self, loss, **kwargs):
        """
        Create loss
        Args:
            @loss (string): select loss function
        """
        pass

    def _create_optimizer(self, optimizer='adam', lr=0.0001, **kwargs ):
        """
        Create optimizer
        Args:
            @optimizer (string): select optimizer function
            @lr (float): learning rate
            @momentum (float): momentum
        """
        
        self.optimizer = None

        # create optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam( self.net.parameters(), lr=lr, amsgrad=True )  
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD( self.net.parameters(), lr=lr, **kwargs ) 
        elif optimizer == 'rprop':
            self.optimizer = torch.optim.Rprop( self.net.parameters(), lr=lr) 
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop( self.net.parameters(), lr=lr)           
        else:
            assert(False)

        self.lr = lr; 
        self.s_optimizer = optimizer

    def _create_scheduler_lr(self, lrsch, **kwargs ):
        
        #MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        #ExponentialLR
        #CosineAnnealingLR

        self.lrscheduler = None

        if lrsch == 'fixed':
            pass           
        elif lrsch == 'step':
            self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs )# step_size=3, gamma=0.1 
        elif lrsch == 'cyclic': 
            self.lrscheduler = netlearningrate.CyclicLR(self.optimizer, **kwargs)
        elif lrsch == 'exp':
            self.lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs ) # gamma=0.99 
        elif lrsch == 'plateau':
            self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs ) # 'min', patience=10
        else:
            assert(False)
        
        self.s_lerning_rate_sch = lrsch

    def adjust_learning_rate(self, epoch ):
        """
        Update learning rate
        """        
        # update
        if self.s_lerning_rate_sch == 'fixed': 
            lr = self.lr
        elif self.s_lerning_rate_sch == 'plateau':
            self.lrscheduler.step( self.vallosses )
            for param_group in self.optimizer.param_groups:
                lr = float(param_group['lr'])
                break            
        else:                    
            self.lrscheduler.step() 
            lr = self.lrscheduler.get_lr()[0]        

        # draw
        self.plotter.plot('lr', 'learning rate', epoch, lr )
 
    def resume(self, resume):
        """
        Resume: optionally resume from a checkpoint
        """ 
        net = self.net.module if self.parallel else self.net
        start_epoch = 0
        prec = 0
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                prec = checkpoint['prec']
                net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                #self.vocabolary.__dict__ =  checkpoint['vocabolary']     
                self.embedding.load_state_dict( checkpoint['embedding'] ) 

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
        net = self.net.module if self.parallel else self.net
        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
                #'vocabolary': self.vocabolary.__dict__,
                'embedding': self.embedding.state_dict()

            }, 
            is_best,
            self.pathmodels,
            filename,
            )
   
    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                                
                self._create_model( checkpoint['arch'], None, False )

                self.embedding.load_state_dict( checkpoint['embedding'] )
                self.net.load_state_dict( checkpoint['state_dict'] )   
                #self.vocabolary.__dict__ =  checkpoint['vocabolary']     
                 
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
                'Model: \n{} \n'.format(
                    self.nameproject,
                    self.s_arch,
                    self.s_loss,
                    self.s_optimizer,
                    self.lr,
                    self.net
                    )
                )

class NeuralNetNLP(NeuralNetAbstractNLP):
    r"""Convolutional Neural Net for classification
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
        super(NeuralNetNLP, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )

 
    def create(self, 
        arch, 
        embedded,
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
            embedded:
            loss (string):
            lr (float): learning rate
            momentum,
            optimizer (string) : 
            lrsch (string): scheduler learning rate
            pretrained (bool)
        """

        cfg_opt= { 'momentum':0.9, 'weight_decay':5e-4 } 
        cfg_scheduler= { 'step_size':100, 'gamma':0.1  }
                    
        super(NeuralNetNLP, self).create( 
            arch, 
            embedded,
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
    
    def _create_model(self, arch, embedding, pretrained):
        """
        Create model
            @arch (string): select architecture
            @embedding:
            @pretrained (bool)
        """    

        self.net = None
        self.encoder = None            

        #if embedding is None:
        self.embedding = nn.Embedding( 74666, 300 ) 
        #elif isinstance(data, torch.Tensor):
        #   self.embedding = embedding
        #else:   
        #   self.embedding = nn.Embedding.from_pretrained( torch.from_numpy( embedding ).float(),  freeze=False )
        
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
        if loss == 'triplet':
            self.criterion = netloss.TripletCosineLoss().cuda()
        else:
            assert(False)
        self.s_loss = loss





