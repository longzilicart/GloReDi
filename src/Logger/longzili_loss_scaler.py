# loss scaler for stablize training

import torch
from collections import deque
from functools import wraps

class LossScaler:
    def __init__(self, max_len=50, scale_factor=2):
        self.history = deque(maxlen=max_len)
        self.scale_factor = scale_factor

    def __call__(self, loss):
        self.history.append(loss.detach())
        max_loss, min_loss = max(self.history), min(self.history)
        if loss.item() > min_loss * self.scale_factor:
            loss = min_loss * self.scale_factor
        return loss



class LossScaler_Wrapper:
    loss_history = deque(maxlen=50)
    scale_factor = 2

    @classmethod
    def wrapper(cls, forward_fn):
        @wraps(forward_fn)
        def wrapped_function(*args, **kwargs):
            loss = forward_fn(*args, **kwargs)
            cls.loss_history.append(loss.detach().item())

            if len(cls.loss_history) == 1:
                return loss

            smooth_median = torch.tensor(cls.get_median(), device=loss.device, requires_grad=True)
            max_loss, min_loss = max(cls.loss_history), min(cls.loss_history)

            if loss.item() > smooth_median.item() * cls.scale_factor:
                loss = torch.tensor(min_loss * cls.scale_factor, device=loss.device, requires_grad=True)
            return loss
        return wrapped_function

    @classmethod
    def get_median(cls):
        return torch.median(torch.tensor(list(cls.loss_history)))



if __name__ == "__main__":
    pass
