from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def per_class_accur_slow_and_stupid(pred, target, n_classes_to_report):
    '''Given
      -  predictions 'pred' with shape b , where K is the overall number of classes, 
      -  and a target vector 'target' with shape b, containing integer class labels,
      - n_classes_to_report, the number of classes to report per-class accuracy for,
      compute the per-class accuracy.

      Use a stupid loop to ensure correctness.
      For every class k <= n_classes_to_report:
        - weight each sample up by n_classes if the sample has target k.
        - compute the weighted accuracy of class k 
      '''
    accurs = []

    # pred = output.argmax(dim=1)

    n_classes = target.max() + 1
    Z = 2 * (n_classes - 1) / n_classes
    # Weighting:
    # We want to weight the in-class samples up by a factor .
    # The factor is w_in = (C - 1) where C is the number of classes.
    # This is because out of C samples, statistically C-1 will 
    # be from another class, and one from this class. In
    # This way the weight offsets the higher frequency of the 
    # other-class samples.
    # 
    # Now we still need to re-scale the weights to get a sensible
    # weighted average: This rescaling is w/Z , 
    # where w in w_in = c-1 and w_other = 1.
    #
    # What is Z?  
    #
    #         accur = 1/N * (#_in * accur_in * w_in/Z + #_o * accur_o * w_o / Z)
    #               = (on average) = 1/N * (N/C * accur_in * (C-1)/Z  
    #                                          + N*(C-1)/C * accur_o * 1 / Z)
    #               = 1/Z*(accur_in * (C-1)/C + accur_o * (C-1)/C)
    #               = (C-1)/C * 1/Z * (accur_in + accur_o)
    #
    #... If both accuracies are 1 (perfect predictor), we want this to be 1.
    #    So Z must be 2 * C / (C-1).
    #   Then at chance accur (acc_in = 1/C, acc_other = (C-1)/C, we predict a random class):
    #     accur =  1/2*(1/C + (C-1)/C) = 1/2. That's nice. 

    for k in range(n_classes_to_report):
        weight = torch.ones_like(target) / Z
        weight[target == k] = (n_classes - 1) / Z

        correct = torch.zeros_like(target, dtype=torch.float32)
        # for non-k samples, we're correct if we don't predict k
        correct[target != k] = (pred[target != k] != k).float()
        # otherwise, we're correct if we predict k
        correct[target == k] = (pred[target == k] == k).float()

        weighted_correct = correct * weight
        accurs.append(weighted_correct.mean() * 100.0) 
    
    return torch.stack(accurs)

def per_class_accuracy(pred, target, n_classes_to_report):
    '''For weighting explanation see slow-and-stupid function above.
    
    Todo: Can profile whether another way to explicitly create the index tensor is faster.'''
    # pred = output.argmax(dim=1)

    N = pred.shape[0]
    n_classes = target.max() + 1
    Z = 2 * (n_classes - 1) / n_classes 

    class_range = torch.arange(n_classes_to_report, device=pred.device).unsqueeze(0)

    indices = target.unsqueeze(1) == class_range

    ids_dim0, ids_dim1 = torch.where(indices)
    _, ids_dim1_n = torch.where(~indices)

    weight = torch.ones(N, n_classes_to_report, device=pred.device, dtype=pred.dtype) / Z
    weight[indices] = (n_classes - 1) / Z

    correct = torch.zeros(N, n_classes_to_report, device=pred.device, dtype=torch.float32)
    pred_expanded = pred.unsqueeze(1).expand(-1, n_classes_to_report)

    # for non-k samples, we're correct if we don't predict k
    correct[~indices] = (pred_expanded[~indices] != class_range[0, ids_dim1_n]).float()
    # otherwise, we're correct if we predict k
    correct[indices] = (pred_expanded[ids_dim0, ids_dim1] == class_range[0, ids_dim1]).float()

    weighted_correct = correct * weight
    accurs = weighted_correct.mean(dim=0) * 100.0
    
    return accurs



def accuracy(output, target, topk=(1,), return_per_class_top_1=False, n_classes_to_report=10):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = [] 
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        if return_per_class_top_1:
            per_class_top_1_slow = per_class_accur_slow_and_stupid(pred[0], target, n_classes_to_report)
            per_class_top_1 = per_class_accuracy(pred[0], target, n_classes_to_report)
            assert torch.allclose(per_class_top_1_slow, per_class_top_1)
            return res, per_class_top_1
        else:
            return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# utils.py

def generate_embeddings_filename(model_type, dataset, model, num_embeddings_per_class, head=False, epoch=None, trial="0"):
    """Generate a filename for embeddings based on parameters."""
    return f'embeddings_{model_type}_{dataset}_{model}_{"num_embeddings_" + str(num_embeddings_per_class) if num_embeddings_per_class != -1 else "all"}{"_head" if head else ""}_ep-{epoch}_trial-{trial}.pt'

