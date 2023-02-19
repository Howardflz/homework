from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Class for a new scheduler.

    Have __init function with parameters and learning rate scheduler function.

    """

    # Non-default arguments should be before default arguments
    def __init__(self, optimizer, step_size, gamma=0.9, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...

        self.gamma = gamma
        self.step_size = step_size

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
        """
        Revise based on LR scheduler.

        After some step_size of epoch, we decay the learning rate by gamma and update the
        new learning rate to the list.

        """

        l = []
        cur = self.last_epoch
        times = cur // self.step_size
        for i in self.base_lrs:
            lr = i * self.gamma**times
            l.append(lr)

        return l
