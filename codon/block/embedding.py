from codon.base import *


class RotaryPositionalEmbedding(BasicModel):
    '''
    旋转位置编码 (Rotary Positional Embedding, RoPE)。
    '''

    def __init__(self, model_dim: int, max_len: int = 128000, base: int = 10000):
        '''
        初始化 RoPE 模块。

        Args:
            model_dim (int): 模型的维度 (或 head_dim)。必须是偶数。
            max_len (int, optional): 预计算位置编码的最大序列长度。默认为 128000。
            base (int, optional): 计算频率的基数。默认为 10000。
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
        将向量分为两半并旋转: [-x2, x1]。
        无论输入是 3D 还是 4D，Split 都是作用在最后一维 (model_dim)。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 旋转后的张量。
        '''
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        '''
        应用旋转位置编码。
        
        自动适配两种输入:
        1. [Batch, Seq_Len, Dim]
        2. [Batch, Head, Seq_Len, Head_Dim]

        Args:
            x (torch.Tensor): 输入张量。
            start_pos (int, optional): 起始位置索引，用于 KV Cache 推理。默认为 0。

        Returns:
            torch.Tensor: 添加了位置信息的张量。
        '''
        ndim = x.ndim
        seq_len = x.shape[-2]

        cos = self.cos_cached[start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[start_pos : start_pos + seq_len, :]
        
        shape = [1] * (ndim - 2) + [seq_len, -1]
        cos = cos.view(*shape)
        sin = sin.view(*shape)

        return (x * cos) + (self._rotate_half(x) * sin)