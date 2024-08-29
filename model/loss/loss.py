import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import math

def nll_loss(output, target):
    return F.nll_loss(output, target)
def binary_cross_entropy(output, target): 
    criterion = nn.CrossEntropyLoss()

    # Ensure the ground truth tensor is of the correct type
    gt = target.long()

    # Compute the loss
    loss = criterion(output, gt)
    return loss


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device = alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha):
    device = y.get_device()
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):

    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num, num_classes, annealing_step))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step: int = 10):

    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    alpha = exp_evidence(output)
    loss = torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss

class EvidentialLoss(nn.Module):
    def __init__(
        self, logvar_eps=1e-4, resi_min=1e-4, resi_max=1e3, num_classes: int = 4
    ) -> None:
        super(EvidentialLoss, self).__init__()
        self.logvar_eps = logvar_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        self.loss_fnc = edl_digamma_loss
        self.num_classes = num_classes

    def _adapt_shape(self, tensor, num_classes: int = 4):
        # print(tensor.shape)
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = torch.reshape(tensor, (-1, num_classes))
        return tensor

    def forward(self, mean: Tensor, target: Tensor, epoch: int):
        mean = self._adapt_shape(mean, num_classes=self.num_classes)
        target = self._adapt_shape(target, num_classes=self.num_classes)

        l = self.loss_fnc(mean, target, epoch, self.num_classes)
        return l
def RecLoss(
    y_hat,
    output,
    target,
    epoch = 0,
    reduction="mean",
    alpha_eps=1e-4,
    beta_eps=1e-4,
    resi_min=1e-4,
    resi_max=1e3,
    ):
    # print(target.shape)
    # One-hot encode the target tensor
    target_onehot = F.one_hot(target, num_classes = y_hat.shape[1]).permute(0, 3, 1, 2)
    identity_loss = nn.L1Loss(reduction = reduction)
    # print(torch.max(y_hat))
    # print(output.shape)
    # print(y_hat.shape)
    l1 = identity_loss(output, y_hat)
    alpha = torch.exp(output)
    output = output.softmax(dim=1)
    evi = EvidentialLoss(num_classes = y_hat.shape[1])
    l2 = evi(output, target_onehot, epoch)  
    loss = 10* l1 + l2  
    return loss