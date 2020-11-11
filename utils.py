import os
import sys
import time
import math

from bisect import bisect_right
import numpy as np
import operator
from functools import reduce
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import socket


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0


def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()

    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = 0
        delta_ops = weight_ops + bias_ops

    global count_ops
    count_ops += delta_ops
    return


def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if str(module).startswith('Sequential'):
        return False
    if is_leaf(module):
        return True
    return False


def cal_multi_adds(model, shape=(2, 3, 32, 32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape)

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)

        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()


class RecallK(nn.Module):
    def __init__(self, K=1):
        super(RecallK, self).__init__()
        self.K = K

    def forward(self, feature_bank, label_bank):
        num_instances = feature_bank.size(0)
        feature_bank_pow2 = torch.pow(feature_bank, 2).sum(dim=1, keepdim=True).expand(num_instances, num_instances)
        distmat = feature_bank_pow2 + feature_bank_pow2.t()
        distmat.addmm_(feature_bank, feature_bank.t(), beta=1, alpha=-2)

        distmat.scatter_(1, torch.arange(num_instances).view(-1, 1).cuda(), distmat.max().item())

        _, predicted = torch.min(distmat, dim=1)

        recall_k = label_bank[predicted].eq(label_bank).sum().item() / num_instances

        return recall_k


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=10, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1-epsilon)
    log_prob = F.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


def adjust_lr(optimizer, epoch, args, eta_min=0.):
    cur_lr = 0.
    if args.lr_type == 'SGDR':
        i = int(math.log2(epoch / args.sgdr_t + 1))
        T_cur = epoch - args.sgdr_t * (2 ** (i) - 1)
        T_i = (args.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (args.init_lr - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def get_data_folder(args, log_dir):
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('ws-W560-G20'):
        if args.dataset.startswith('CIFAR'):
            data_folder = '/home/ws/winycg/dataset'
        elif args.dataset == 'tinyimagenet':
            data_folder = '/home/ws/winycg/dataset/tiny-imagenet-200/'
        elif args.dataset == 'CUB200':
            data_folder = '/home/ws/winycg/dataset/CUB_200_2011/'
        elif args.dataset == 'STANFORD120':
            data_folder = '/home/ws/winycg/dataset/standford-dogs/'
        elif args.dataset == 'MIT67':
            data_folder = '/home/ws/winycg/dataset/MIT-indoor-67/'
        else:
            raise ValueError('unknown dataset')
        args.checkpoint_dir = '/home/ws/winycg/self_kd_checkpoint/'
    elif hostname.startswith('ubuntu'):
        if args.dataset == 'imagenet':
            data_folder = '/dev/shm/'
        if args.dataset.startswith('CIFAR'):
            data_folder = '/home/user/winycg/dataset'
        args.checkpoint_dir = '/home/user/winycg/self_kd_checkpoint/'
    elif hostname.startswith('winycgv1'):
        if args.dataset.startswith('CIFAR'):
            data_folder = '/home/user/hhd/dataset/'
        elif args.dataset == 'tinyimagenet':
            data_folder = '/home/user/hhd/dataset/tiny-imagenet-200'
        elif args.dataset == 'CUB200':
            data_folder = '/home/user/hhd/dataset/CUB_200_2011/'
        elif args.dataset == 'STANFORD120':
            data_folder = '/home/user/hhd/dataset/standford-dogs/'
        elif args.dataset == 'MIT67':
            data_folder = '/home/user/hhd/dataset/MIT-indoor-67/'
        elif args.dataset == 'imagenet':
            data_folder = '/home/user/dataset/imagenet/'
        else:
            raise ValueError('unknown dataset')
        args.checkpoint_dir = '/home/user/hhd/self_kd_checkpoint/'
    else:
        data_folder = args.data_folder

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    return data_folder


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=64, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, in_dim=64, out_dim=128):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.l2norm = Normalize(2)
        self.embed_module = Embed(dim_in=in_dim, dim_out=out_dim)

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = self.embed_module(features)
        features = self.l2norm(features)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        #anchor_labels = labels[batch_size//2:, :]
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_feature = features
        contrast_feature = features

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss



def mixup_data(x, y, args=None, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam, index

    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    if lam == 0.5:
        cx = np.random.randint(W + 1 - cut_w)
        cy = np.random.randint(H + 1 - cut_h)
        bbx1 = np.clip(cx, 0, W)
        bby1 = np.clip(cy, 0, H)
        bbx2 = np.clip(cx + cut_w, 0, W)
        bby2 = np.clip(cy + cut_h, 0, H)
    else:
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, args=None, alpha=1.):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    batch_size = x.size()[0]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    if alpha > 0:
        rand_index = torch.randperm(x.size()[0]).cuda()
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam, rand_index
        '''
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:batch_size // 2, :, bbx1:bbx2, bby1:bby2] = x[batch_size // 2:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, lam
        '''
    else:
        cutmix_x = x[:batch_size // 2, :, :, :].clone()
        cutmix_x[:, :, bbx1:bbx2, bby1:bby2] = x[batch_size // 2:, :, bbx1:bbx2, bby1:bby2]
        return cutmix_x

def cutmix_data_lam(x, y, args=None, lam=0.5):
    batch_size = x.size()[0]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    cutmix_x = x[:batch_size // 2, :, :, :].clone()
    cutmix_x[:, :, bbx1:bbx2, bby1:bby2] = x[batch_size // 2:, :, bbx1:bbx2, bby1:bby2]
    return cutmix_x, lam


def index_cutmix_data(x, y, index, args=None, alpha=1.):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    batch_size = x.size()[0]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    if alpha > 0:
        rand_index = index
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam, rand_index
        '''
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:batch_size // 2, :, bbx1:bbx2, bby1:bby2] = x[batch_size // 2:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, lam
        '''
    else:
        cutmix_x = x[:batch_size // 2, :, :, :].clone()
        cutmix_x[:, :, bbx1:bbx2, bby1:bby2] = x[batch_size // 2:, :, bbx1:bbx2, bby1:bby2]
        return cutmix_x

