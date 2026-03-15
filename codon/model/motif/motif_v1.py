from codon.base import *

from codon.block.codebook  import LookupFreeQuantization, LookupFreeQuantizationOutput
from codon.block.attention import MultiHeadAttention, AttentionOutput
from codon.block.embedding import InterleavedRotaryEmbedding
from codon.model.resnet    import ResNet

from dataclasses import dataclass

@dataclass
class MotifV1EncoderOutput: ...



class MotifV1Encoder(BasicModel):
    def __init__(
        self,
        in_features: int = 3,
        patch_size: int = 8,
        latent_dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        codebook_dim: int = 18,
        entropy_weight: float = 0.1,
        commitment_weight: float = 0.25,
        diversity_gamma: float = 1.0,
        rope_emb: InterleavedRotaryEmbedding = None
    ):
        super().__init__()
        self.in_features = in_features
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.codebook_dim = codebook_dim
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_weight
        self.diversity_gamma = diversity_gamma

        self.resnet = ResNet.auto_build(
            input_shape=(in_features, patch_size, patch_size),
            output_shape=(latent_dim,)
        )

        self.attn = MultiHeadAttention(
            hidden_size=latent_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
            use_gate=True,
            is_causal=False
        )

        self.codebook = LookupFreeQuantization(
            latent_dim=latent_dim,
            codebook_dim=codebook_dim,
            entropy_weight=entropy_weight,
            commitment_weight=commitment_weight,
            diversity_gamma=diversity_gamma
        )

        self.rope_emb = rope_emb if isinstance(rope_emb, InterleavedRotaryEmbedding) and rope_emb.num_axes >= 2 else InterleavedRotaryEmbedding(
            model_dim=latent_dim,
            num_axes=2
        )
    
    def forward(
        splited_image: torch.Tensor,
        grid_shape: tuple,
        rope_emb: InterleavedRotaryEmbedding = None
    ) -> MotifV1EncoderOutput:
        '''
        splited_image (num_patches, channels, patch_size, patch_size)
        grid_shape (num_patches_h, num_patches_w)
        rope_emb 2dInterleavedRotaryEmbedding
        '''
        pass


class MotifV1Decoder(BasicModel):
    def __init__(self):
        super().__init__()
