'''Copied from https://github.com/hazdzz/RMSNorm/blob/main/norm.py. 
    See license in rmsnorm_LICENSE.'''

import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm
    




class RMSNorm2d(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size],  
                 eps: float = 1e-5, bias: bool = False) -> None:
        '''
        :param normalized_shape: feature dimensions size of the input tensor
        '''
        super(RMSNorm2d, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape) 
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((*self.normalized_shape, 1, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty((*self.normalized_shape, 1, 1)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-3, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm
    


