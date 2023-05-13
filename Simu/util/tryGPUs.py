# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 10:35
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numbered:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numbered:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
