import torch
import torch.nn as nn

from typing import Callable, Any, Iterator


class BasicModel(nn.Module):
    '''
    Base class for all models, providing common functionality like gradient checkpointing and parameter counting.
    '''
    def __init__(self):
        '''
        Initialize the BasicModel.
        '''
        super(BasicModel, self).__init__()
        self.gradient_checkpointing: bool = False
    
    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device where the model parameters are located.
                          Returns 'cpu' if the model has no parameters.
        '''
        try: return next(self.parameters()).device
        except StopIteration: return torch.device('cpu')
    
    def set_checkpoint(self, value:bool) -> None:
        '''
        Enable or disable gradient checkpointing for the model and its sub-modules.

        Args:
            value (bool): True to enable gradient checkpointing, False to disable.
        '''
        self.gradient_checkpointing = value
        for model in self.modules():
            if isinstance(model, BasicModel) and model is not self:
                model.gradient_checkpointing = value

    def checkpoint(self, function:Callable, *args, **kwargs) -> Any:
        '''
        Apply gradient checkpointing to a function if enabled and in training mode.

        Args:
            function (Callable): The function to be checkpointed.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The output of the function.
        '''
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                function, *args, use_reentrant=False, **kwargs
            )
        return function(*args, **kwargs)
    
    def get_params(self, trainable_only:bool=False) -> Iterator[torch.nn.Parameter]:
        '''
        Get an iterator over the model parameters.

        Args:
            trainable_only (bool, optional): If True, only yield parameters that require gradients.
                                             Defaults to False.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over the model parameters.
        '''
        if trainable_only:
            return (p for p in self.parameters() if p.requires_grad)
        return self.parameters()
    
    def count_params(self, trainable_only:bool=False) -> int:
        '''
        Count the number of parameters in the model.

        Args:
            trainable_only (bool, optional): If True, count only trainable parameters.
                                             Defaults to False.

        Returns:
            int: The total number of parameters.
        '''
        return sum(p.numel() for p in self.get_params(trainable_only))
