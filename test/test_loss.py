

import sys
sys.path.append('../')
import pytest

import torch
import torch.nn as nn
from torchlib import loss


def test_triplet_loss():

    dim=300; batch=5
    s1 = torch.randn( batch, dim )
    s2 = torch.randn( batch, dim )
    t1 = torch.randn( batch, dim )

    print(s1.shape)
    print(s2.shape)
    print(t1.shape)

    l = loss.tripletCosineLoss(s1, s2, t1)
    print(l)

test_triplet_loss()
