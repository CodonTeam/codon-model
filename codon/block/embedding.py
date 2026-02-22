from codon.base import *


class RotaryPositionalEmbedding(BasicModel):
    '''
    Rotary Positional Embedding (RoPE).
    '''

    def __init__(self, model_dim: int, max_len: int = 128000, base: int = 50000):
        '''
        Initialize the RoPE module.

        Args:
            model_dim (int): The dimension of the model (or head_dim). Must be even.
            max_len (int, optional): Maximum sequence length for pre-computing position encodings. 
                                     Defaults to 128000.
            base (int, optional): Base for computing frequencies. Defaults to 50000.
        '''
        super(RotaryPositionalEmbedding, self).__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, model_dim, 2).float() / model_dim))
        
        t = torch.arange(max_len, dtype=torch.float)
        
        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Split the vector into two halves and rotate them: [-x2, x1].
        The split operation is performed on the last dimension (model_dim),
        regardless of whether the input is 3D or 4D.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        '''
        Apply rotary positional encoding.
        
        Automatically adapts to two types of inputs:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int, optional): Starting position index for KV Cache inference.
                                       Defaults to 0.

        Returns:
            torch.Tensor: Tensor with positional information added.
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]

        cos = self.cos_cached[start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[start_pos : start_pos + seq_len, :]
        
        shape = [1] * (ndim - 2) + [seq_len, -1]
        cos = cos.view(*shape)
        sin = sin.view(*shape)

        return (x * cos) + (self._rotate_half(x) * sin)
