import torch
import torch.nn as nn
from torch.nn import functional as F

class MLCrossEntropy(nn.Module):
    def __init__(self,):
        super(MLCrossEntropy, self).__init__()
        # self.s0 = torch.tensor(0.3)
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction="none")
        self.loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

    def forward(self, logit, target, mask=None):
        logit = logit.view(-1, logit.shape[-1])
        target = target.view(-1, target.shape[-1])
        
        loss = - logit * target
        mask = target.gt(0).float()
        mask = torch.sum(mask, -1)
        mask1 = torch.where(mask==0, torch.tensor(1, dtype=torch.float32, device=mask.device), mask)
        loss = torch.sum(loss, -1) / mask1
        mask2 = mask.gt(0).float()
        loss = torch.sum(loss) / torch.sum(mask2)
        return loss