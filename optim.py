
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
                        _dispatch_sqrt, _default_to_fused_or_foreach, _capturable_doc,
                        _differentiable_doc, _foreach_doc, _fused_doc, _maximize_doc)
from typing import List, Optional
import torch
from torch import Tensor

class MyAdam(Optimizer):
    def __init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, 
                 *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        