import torch
import torch.nn as nn

from typing import Callable, Any, Iterator


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.gradient_checkpointing: bool = False
    
    @property
    def device(self) -> torch.device:
        try: return next(self.parameters()).device
        except StopIteration: return torch.device('cpu')
    
    def set_checkpoint(self, value:bool) -> None:
        self.gradient_checkpointing = value
        for model in self.modules():
            if isinstance(model, BasicModel) and model is not self:
                model.gradient_checkpointing = value

    def checkpoint(self, function:Callable, *args, **kwargs) -> Any:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                function, *args, use_reentrant=False, **kwargs
            )
        return function(*args, **kwargs)
    
    def get_params(self, trainable_only:bool=False) -> Iterator[torch.nn.Parameter]:
        if trainable_only:
            return (p for p in self.parameters() if p.requires_grad)
        return self.parameters()
    
    def count_params(self, trainable_only:bool=False) -> int:
        return sum(p.numel() for p in self.get_params(trainable_only))