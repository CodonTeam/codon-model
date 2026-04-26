from codon.block import BasicManifoldLinear, RiemannianManifoldLinear
from codon.base  import *

from typing import Union


def to_manifold(model: Union[nn.Module, BasicModel]): ...

def to_linear(model: Union[nn.Module, BasicModel]): ...