from xml.dom.domreg import well_known_implementations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES
from .utils import weighted_loss

 


@weighted_loss
def n_pair_loss(weight,target):
    #d, n = weight.size()
    a = F.cosine_similarity(target, weight, dim=0, eps=1e-08)
    a = torch.exp(a-1)
    a = torch.sum(a)
    #a = a * torch.exp(torch.tensor([2.0]).cuda()) / n
    loss =  torch.log(a)
    return loss
@LOSSES.register_module
class NPairLoss(nn.Module):
    def __init__(self, loss_weight):
        super(NPairLoss, self).__init__()
        self.loss_weight = loss_weight
    def forward(self,
                weight):
        loss = 0
        d, n = weight.size()
        for i in range(n):
            target = weight.data[:,i].reshape(d,1)
            target = target.expand(d,n)
            loss += n_pair_loss(weight, target)
        loss_npair = self.loss_weight * loss / n
        return loss_npair
