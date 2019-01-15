import torch
import torch.nn.functional as F
import torch.nn as nn

class MaskNLLLoss(nn.Module):
    def __init__(self):
        super(MaskNLLLoss, self).__init__()        
    def forward(self, inp, target, mask ):
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = crossEntropy.masked_select(mask).mean()
        return loss

class TripletCosineLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-6 ):
        super(TripletCosineLoss, self).__init__()    
        self.margin = margin
        self.eps = eps    
    def forward(self, s1, s2, t1 ):        
        coss1s2 = F.cosine_similarity(s1, s2, dim=1, eps=self.eps)
        coss1t1 = F.cosine_similarity(s1, t1, dim=1, eps=self.eps)    
        triplet_loss = F.relu( self.margin - coss1s2 + coss1t1  ).mean()
        return triplet_loss


class Accuracy(nn.Module):
    def __init__(self, margin=0.0, eps=1e-6 ):
        super(Accuracy, self).__init__()
        self.margin = margin
        self.eps=eps
    def forward(self, s1, s2, t1):
        coss1s2 = F.cosine_similarity(s1, s2, dim=1, eps=self.eps)
        coss1t1 = F.cosine_similarity(s1, t1, dim=1, eps=self.eps)   
        pred = (coss1s2 - coss1t1 - self.margin).cpu().data    
        pred = (pred > 0).float().sum() / coss1s2.shape[0]
        
        return pred

