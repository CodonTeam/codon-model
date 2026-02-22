import torch.nn.functional as F
import math

from codon.base  import *
from dataclasses import dataclass
from typing      import Optional, Tuple

from torch.nn.attention import SDPBackend, sdpa_kernel

from codon.block.embedding import RotaryPositionalEmbedding


@dataclass
class AttentionOutput:
    '''
    Output of the attention mechanism.

    Args:
        output (torch.Tensor): The output tensor from the attention mechanism.
        attention_weights (Optional[torch.Tensor], optional): Attention weights. Defaults to None.
        past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): 
            Cached key and value tensors for autoregressive generation. Defaults to None.
    '''
    output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def apply_attention(
    query_states: torch.Tensor, 
    key_states: torch.Tensor, 
    value_states: torch.Tensor, 
    attention_mask: torch.Tensor = None, 
    output_attentions: bool = False,
    is_causal: bool = None,
    dropout: float = 0.0
) -> AttentionOutput:
    ''' 
    Compute scaled dot-product attention.

    Args:
        query_states (torch.Tensor): Query states tensor.
        key_states (torch.Tensor): Key states tensor.
        value_states (torch.Tensor): Value states tensor.
        attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
        is_causal (bool, optional): Whether to apply a causal mask. Defaults to None.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        AttentionOutput: Object containing attention output and optional weights.
    '''
    
    if attention_mask is not None:
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()
        
        if attention_mask.max() <= 1.0:
            attention_mask = torch.where(attention_mask == 0, float('-inf'), 0.0)
        
    if is_causal:
        tgt_len = query_states.size(-2)
        src_len = key_states.size(-2)
        
        causal_mask = torch.tril(
            torch.ones((tgt_len, src_len), device=query_states.device, dtype=query_states.dtype)
        ).view(1, 1, tgt_len, src_len)
        
        causal_mask = torch.where(causal_mask == 0, float('-inf'), 0.0)
        
        if attention_mask is not None:
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask
        
        is_causal = False
        
    if not output_attentions:
        if attention_mask is None and is_causal is None: 
            is_causal = True
        
        try:
            with sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION
            ]):
                output = F.scaled_dot_product_attention(
                    query_states, 
                    key_states, 
                    value_states, 
                    attn_mask=attention_mask,
                    is_causal=is_causal,
                    dropout_p=dropout
                )
            return AttentionOutput(output=output, attention_weights=None)
        except RuntimeError: 
            pass
    # Manual Fallback Path
    d_k = query_states.size(-1)
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(d_k)

    if attention_mask is not None:
        scores = scores + attention_mask
    attention_weights = torch.softmax(scores, dim=-1)
    
    if dropout > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout)
    output = torch.matmul(attention_weights, value_states)
    
    return AttentionOutput(output=output, attention_weights=attention_weights)


class MultiHeadAttention(BasicModel):
    ''' 
    Multi-Head Attention module.
    Supports Grouped Query Attention (GQA), QK Normalization, and Gating mechanisms.

    Args:
        hidden_size (int): Size of the hidden layer.
        num_heads (int): Number of attention heads.
        num_kv_heads (int, optional): Number of key/value heads for GQA. 
                                      If None, defaults to num_heads.
        use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. 
                                      Defaults to True.
        use_gate (bool, optional): Whether to apply a gating mechanism. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        is_causal (bool, optional): Whether to apply a causal mask. 
                                    Defaults to True (for Decoder architectures).
    '''
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads=None,
        use_qk_norm=True,
        use_gate=False,
        dropout=0.1,
        is_causal=True
    ):
        super(MultiHeadAttention, self).__init__()

        if num_kv_heads is None: num_kv_heads = num_heads

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_heads  = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim  = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate
        self.dropout = dropout
        self.is_causal = is_causal
        
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        rotary_emb: RotaryPositionalEmbedding = None,
        rotary_pos: int = 0,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> AttentionOutput:
        ''' 
        Perform forward pass of Multi-Head Attention.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            kv_states (torch.Tensor, optional): Hidden states for keys/values. 
                                                If None, uses hidden_states. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask (usually Padding Mask). 
                                                     Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. 
                                                Defaults to False.
            rotary_emb (RotaryPositionalEmbedding, optional): Rotary positional embedding module. 
                                                              Defaults to None.
            rotary_pos (int, optional): Starting position for rotary embedding. Defaults to 0.
            past_key_value (tuple[torch.Tensor, torch.Tensor], optional): Past key-value cache. 
                                                                          Defaults to None.
            use_cache (bool, optional): Whether to use KV cache. Defaults to False.
        
        Returns:
            AttentionOutput: Object containing output, attention weights, and KV cache.
        '''
        
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        Q = self.q_proj(hidden_states)
        K = self.k_proj(kv_states)
        V = self.v_proj(kv_states)

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        if rotary_emb is not None:
            Q = rotary_emb(Q, start_pos=rotary_pos)
            K = rotary_emb(K, start_pos=rotary_pos)
        
        current_key_value = None
        if use_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                K = torch.cat((past_k, K), dim=2)
                V = torch.cat((past_v, V), dim=2)
            current_key_value = (K, V)

        kv_seq_len_total = K.shape[2]

        if self.num_kv_queries > 1:
            # [B, H_kv, 1, L, D] -> [B, H_kv, G, L, D]
            K = K[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            V = V[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            
            K = K.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            V = V.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            
        attn_output = apply_attention(
            Q, K, V, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions,
            is_causal=self.is_causal,
            dropout=self.dropout if self.training else 0.0
        )

        output = attn_output.output
        attention_weights = attn_output.attention_weights
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        output = self.o_proj(output)

        if self.use_gate: output = output * G

        return AttentionOutput(
            output=output,
            attention_weights=attention_weights,
            past_key_value=current_key_value
        )
