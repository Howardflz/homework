from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import math


class CustomLRScheduler(_LRScheduler):
    """
    Class for a new scheduler.

    Have __init function with parameters and learning rate scheduler function.

    """

    # Non-default arguments should be before default arguments
    def __init__(self, optimizer, T_0, T_mult=1, lr_min=0, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...

        self.T_0 = T_0
        self.lr_min = lr_min
        self.restart_idx = 0
        self.T_mult = T_mult
        self.T_cur = T_0

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
        """
        Revise based on LR scheduler.

        We keep track of current epoch and the number since most recent restart index,
        when it reaches the T.cur, we update several parameters and begin the next period.
        During the whole learning process, we use cosine method with restart to learn.

        """

        T_cur = self.last_epoch - self.restart_idx
        if T_cur >= self.T_cur:
            self.restart_idx = self.last_epoch
            self.T_cur = self.T_cur * self.T_mult
            self.T_cur = math.ceil(self.T_cur)
            self.T_0 = self.T_cur

        lr_store_list = []

        for i in self.base_lrs:
            dif = i - self.lr_min
            cos_dif = dif * (1 + math.cos(math.pi * T_cur / self.T_cur)) / 2
            lr = self.lr_min + cos_dif
            lr_store_list.append(lr)

        return lr_store_list
