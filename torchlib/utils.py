import os
import shutil
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = os.path.join(path, 'model_best.pth.tar')
    torch.save(state, filename)

    #torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

def resumecheckpoint(resume, net, optimizer):
    """Optionally resume from a checkpoint"""
    start_epoch = 0
    prec = 0
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            prec = checkpoint['prec']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return start_epoch, prec

