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
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
            1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
            不用加激活函数，尤其是不能加sigmoid或者softmax！预测
            阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
            本文。
        """
        y_true = target.view(-1, target.shape[-1])
        y_pred = logit.view(-1, logit.shape[-1])
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        loss = loss.mean()
        return loss
        # logit = logit.view(-1, logit.shape[-1])
        # target = target.view(-1, target.shape[-1])
        
        # loss = - logit * target
        # mask = target.gt(0).float()
        # mask = torch.sum(mask, -1)
        # mask1 = torch.where(mask==0, torch.tensor(1, dtype=torch.float32, device=mask.device), mask)
        # loss = torch.sum(loss, -1) / mask1
        # mask2 = mask.gt(0).float()
        # loss = torch.sum(loss) / torch.sum(mask2)
        # return loss

class FocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):
        sigmoid_pred = pred.sigmoid()
        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        loss = (target == 1) * self.alpha * torch.pow(1. - sigmoid_pred, self.gamma) * log_sigmoid

        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)
        loss += (target == 0) * (1 - self.alpha) * torch.pow(sigmoid_pred, self.gamma) * log_sigmoid_inv

        return -loss