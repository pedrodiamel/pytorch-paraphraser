
import torch
import torch.nn.functional as F
import torch.nn as nn

from nltk.translate.bleu_score import corpus_bleu

class MaskNLLLoss(nn.Module):
    def __init__(self):
        super(MaskNLLLoss, self).__init__()    

    def forward(self, inp, target, mask ):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = crossEntropy.masked_select(mask).mean()
        return loss, nTotal.item()


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






class Bleu(nn.Module):
    def __init__(self ):
        super(Bleu, self).__init__()


    def forward(self, s, h):



        return 





# import math
# import numpy as np
# from collections import Counter

# def bleu_stats(hypothesis, reference):
#     """Compute statistics for BLEU."""
#     stats = []
#     stats.append(len(hypothesis))
#     stats.append(len(reference))
#     for n in range(1, 5):
#         s_ngrams = Counter(
#             [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
#         )
#         r_ngrams = Counter(
#             [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
#         )
#         stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
#         stats.append(max([len(hypothesis) + 1 - n, 0]))
#     return stats

# def bleu(stats):
#     """Compute BLEU given n-gram statistics."""
#     if len(filter(lambda x: x == 0, stats)) > 0:
#         return 0
#     (c, r) = stats[:2]
#     log_bleu_prec = sum(
#         [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
#     ) / 4.
#     return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


# def get_bleu(hypotheses, reference):
#     """Get validation BLEU score for dev set."""
#     stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
#     for hyp, ref in zip(hypotheses, reference):
#         stats += np.array(bleu_stats(hyp, ref))
#     return 100 * bleu(stats)
