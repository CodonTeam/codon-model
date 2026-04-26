from codon.block import BasicLoRA
from codon.base  import *

from typing import Union, Optional


def inject(
    model: Union[nn.Module, BasicModel],
    layer: Optional[Union[str, list[str]]] = None
):
    layer = layer if layer is not None else 'linear'
    if isinstance(layer, str): layer = [layer]
    