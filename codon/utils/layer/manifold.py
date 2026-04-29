from codon.block import BasicManifoldLinear, RiemannianManifoldLinear
from codon.base  import *

from codon.exp.block.manifold import ExpManifoldLinear

from typing import Union


def to_manifold(model: Union[nn.Module, 'BasicModel'], use_exp: bool = False) -> Union[nn.Module, 'BasicModel']:
    '''
    Recursively replaces all nn.Linear layers in the model with RiemannianManifoldLinear 
    or ExpManifoldLinear. Preserves the original weights and biases.

    Args:
        model (Union[nn.Module, BasicModel]): The model to be converted.
        use_exp (bool): If True, uses ExpManifoldLinear (heterogeneous kappa/lambda). 
                        Otherwise, uses RiemannianManifoldLinear.

    Returns:
        The converted model (modified in-place).
    '''
    TargetClass = ExpManifoldLinear if use_exp else RiemannianManifoldLinear
    
    for name, child in model.named_children():
        if type(child) is nn.Linear:
            new_layer = TargetClass(
                in_features=child.in_features,
                out_features=child.out_features
            )
            
            with torch.no_grad():
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
            
            setattr(model, name, new_layer)
        else:
            to_manifold(child, use_exp)
            
    return model


def to_linear(model: Union[nn.Module, 'BasicModel']) -> Union[nn.Module, 'BasicModel']:
    '''
    Recursively replaces all instances of BasicManifoldLinear (and its subclasses) 
    with standard nn.Linear layers. Preserves the manifold anchors as linear weights.

    Args:
        model (Union[nn.Module, BasicModel]): The model to be converted.

    Returns:
        The converted model (modified in-place).
    '''
    for name, child in model.named_children():
        if isinstance(child, BasicManifoldLinear):
            new_layer = nn.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=True
            )
            
            with torch.no_grad():
                new_layer.weight.data.copy_(child.weight.data)
                new_layer.bias.data.copy_(child.bias.data)
            
            setattr(model, name, new_layer)
        else:
            to_linear(child)
            
    return model