
# STD MODULE
import os
import numpy as np
import cv2
import random

# TORCH MODULE
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

# LOCAL MODULE
from torchlib.datasets.dataset  import TxtTripletDataset
from torchlib.neuralnet import NeuralNetNLP

from argparse import ArgumentParser
import datetime

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--vocabulary', type=str, metavar='NAME',
                        help='(default: none)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')


    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                        help='mini-batch size (default: 256)')
    parser.add_argument('-n', '--nbatch', default=30, type=int, metavar='N', 
                        help='number of batch (default: 30)')


    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', type=str, metavar='NAME',
                        help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                        help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Parallel')

    parser.add_argument('--name-dataset', default='txt', type=str,
                        help='name dataset')


    return parser


def main():
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();
    random.seed( args.seed )
    cudnn.benchmark = True
    
    print('Baseline nlp paragrap paraphrase {}!!!'.format(datetime.datetime.now()))
    print('\nArgs:')
    [ print('\t* {}: {}'.format(k,v) ) for k,v in vars(args).items() ]
    print('')
    
    
    # datasets
    # training dataset
    dataset = TxtTripletDataset(
        pathname=args.data,  
        pathvocabulary=args.vocabulary,
        nbatch=args.nbatch,
        batch_size=args.batch_size
    )
        
        
    print('Load datset')
    print( len(dataset) )
    #print('Train: ', len(train_data))
    #print('Val: ', len(val_data))


    network = NeuralNetNLP(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=args.parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu
        )

    network.create( 
        arch=args.arch, 
        embedded=dataset.voc.embeddings,
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        )    
    
    # resume model
    if args.resume:
        network.resume( os.path.join(network.pathmodels, args.resume ) )

    # print neural net class
    print('Load model: ')
    print(network)
    
    # training neural net
    network.fit( dataset, dataset, args.epochs, args.snapshot )
    
    print("Optimization Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    main()

