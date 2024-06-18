# Author: Changmao Cheng <chengchangmao@megvii.com>

import torch
from torch.nn.functional import cross_entropy


class TransferLoss:

    def __init__(self,beta: float = 1.0) -> None:
        self.beta=beta

    def __call__(self, mu: torch.Tensor, labels: torch.LongTensor, mean: torch.LongTensor) -> torch.Tensor:
        """call function as forward

        Args:
            logits (torch.Tensor): The predicted logits before softmax with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """
        
        loss_vb=None
        for idx,label in enumerate(labels):
            other= torch.full((mean.shape[0],),True,dtype=bool)
            other[label]=False
            other_means = mean[other]
            loss_b=mu[idx]*other_means
            loss_b=torch.sum(loss_b,1)
            loss_b=torch.mean(loss_b)
            if loss_vb:
                loss_vb=loss_vb+loss_b
            else:
                loss_vb=loss_b
        loss_vb=loss_vb/len(labels)
        
        matching_mean = mean[labels] 
        loss_va= torch.pow(mu-matching_mean,2)
        loss_va = torch.sum(loss_va, 1)
        loss_va=torch.mean(loss_va)
        loss = loss_va-self.beta*torch.mean(loss_vb)
        return loss