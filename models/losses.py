import torch
import torch.nn as nn
from torch.nn import functional as F

class MLCrossEntropy(nn.Module):
    def __init__(self,):
        super(MLCrossEntropy, self).__init__()
        # self.s0 = torch.tensor(0.3)
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction="none")
        self.loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

    def forward(self, logit, target, mask):
        logit = logit.view(-1, logit.shape[-1])
        target = target.view(-1, target.shape[-1])
        mask = mask.view(-1)
        loss = self.loss(logit, target)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss