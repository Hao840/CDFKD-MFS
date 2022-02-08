import torch


class InstanceMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()


class FeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def remove(self):
        self.hook.remove()


def multi_forward(models, x, out_feature=False):
    feats = []
    logits = []
    for model in models:
        if out_feature:
            logit, feat = model(x, out_feature)
            logits.append(logit)
            feats.append(feat)
        else:
            logit = model(x, out_feature)
            logits.append(logit)
    if out_feature:
        return logits, feats
    else:
        return logits
