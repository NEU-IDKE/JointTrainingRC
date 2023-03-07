import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCELoss
from torch.nn import CrossEntropyLoss


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
        self.loss = BCELoss()
    
    def forward(self, logits, labels):
        # logits = logits[labels!=0]
        labels[labels!=0] = 1
        labels[labels==0] = 0
        loss = self.loss(logits.view(-1), labels.float())
        return loss
        
class ClassfyLoss(nn.Module):
    def __init__(self, a):
        super(ClassfyLoss, self).__init__()
        self.loss1 = CrossEntropyLoss()
        self.loss2 = TripletLoss()
        self.a = a
    
    def forward(self, logits, labels):

        logits = logits[labels!=0]
        labels = labels[labels!=0]
        if not labels.cpu().tolist():
            return 0
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        return self.a*loss1+(1-self.a)*loss2

    
class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        inputs = inputs[targets!=0]
        targets = targets[targets!=0]
        if not targets.cpu().tolist():
            return 0
        n = inputs.size(0)  # batch_size
        # print(targets)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         print(targets)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if torch.equal(mask[i] == 0, torch.zeros_like(mask[i]) == 1):
                dist_an.append(Variable(torch.tensor([1000],dtype=torch.float), requires_grad=True).cuda())
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class N_Loss(nn.Module):
    def __init__(self):
        super(N_Loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        output = torch.mm(output, output.t().contiguous())
        # mask = torch.tril(torch.ones_like(output))
        # output[mask == 1] = 0
        # output = output.view(-1)
        similrity = []
        n = len(target)
        label = []
        for i in range(n):
            for j in range(i+1, n):
                similrity.append(output[i][j].unsqueeze(0))
                if target[i] == target[j]:
                    label.append(1)
                else:
                    label.append(0)
        similrity = torch.cat(similrity)
        label = torch.tensor(label, dtype=torch.long).cuda()
        assert len(similrity) == len(label)
        assert similrity.requires_grad == True
        return self.bce_loss(similrity, label.float())

class  InfoNce_Loss(nn.Module):
    def __init__(self, t=0.05):
        super(InfoNce_Loss,self).__init__()
        self.t = t

    def forward(self, inputs, labels):
        inputs = inputs[labels!=0]
        labels = labels[labels!=0]
        n = inputs.shape[0]
        norm_emb = F.normalize(inputs, dim=1, p=2)
        sim_score = torch.matmul(norm_emb, norm_emb.transpose(0,1))
        sim_score = sim_score - (torch.eye(n) * 1e12).cuda()
        sim_score = sim_score / self.t
        # infonce loss 的分母提前计算存储（batch_size,）
        base_score = torch.sum(torch.exp(sim_score), dim=1, keepdims=True)
        # mask matrix
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).float() - torch.eye(n).cuda()
        sim_score = torch.exp(sim_score)/base_score
        loss_score = -torch.log(torch.masked_select(sim_score, mask==1))
        loss = torch.mean(loss_score)
        return loss




