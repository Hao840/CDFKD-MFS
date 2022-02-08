import torch.nn.functional as F


def kdloss(y, teacher_scores, T=1., reduction='batchmean'):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, reduction=reduction) * (T * T)
    return l_kl
