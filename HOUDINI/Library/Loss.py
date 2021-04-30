import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import math


def cox_ph_loss_sorted(log_h: Tensor,
                       events: Tensor,
                       reduction: str = 'none',
                       eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    if reduction == 'none':
        return - log_h.sub(log_cumsum_h).mul(events)
    else:
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def cox_ph_loss(log_h: Tensor,
                event_dur: Tensor,
                # events: Tensor,
                # durations: Tensor,
                reduction: str = 'none',
                eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.

    https://github.com/havakv/pycox
    """
    events = event_dur[:, 0]
    durations = event_dur[:, 1]
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, reduction, eps)
