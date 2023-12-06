"""
Attention on Attention Implementation
This is a practice implementation after randomly finding it on Lucidrain's repo, I'm implementing the model architecture just for practice!

Basically the architecture is: x => q, k, v -> multihead attn with residual q -> concat -> 2 linear projects ->sigmoid -> mult -> add -> norm -> ffn -> add -> norm with residual of first add and norm

"""


import torch
from einops import rearrange
from torch import einsum, nn
from zeta.nn import FeedForward

# helpers


def exists(val):
    return val is not None


# normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())



class AoA(nn.Module):
    """Attention on Attention
    
    Args:
        dim (int): Input dimension
        heads (int): Number of heads
        dim_head (int): Dimension of each head
        dropout (float): Dropout
        ff_mult (int, optional): Feedforward multiplier. Defaults to 4.
        depth (int, optional): Depth of the model. Defaults to 1.
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model
        
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        ff_mult: int = 4,
        depth_aoa: int = 1,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.scale = dim_head**-0.5
        self.ff_mult = ff_mult
        self.depth_aoa = depth_aoa
        self.ff_mult = 4

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

        # MultiHeadAttention -> returns a tuple of (attn, attn_weights)
        self.attn = nn.MultiheadAttention(
            self.dim,
            self.heads,
            self.dropout,
        )

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Feedforward
        self.ff = FeedForward(
            dim,
            dim,
            4,
        )
        
        self.proj1 = nn.Linear(dim * 2, dim)
        self.proj2 = nn.Linear(dim * 2, dim)

    
    def forward(self, x: torch.Tensor):
        """Forward pass of the model

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # Linear projection from x -> k, v, q
        k, v, q = self.k_proj(x), self.v_proj(x), self.q_proj(x)

        for _ in range(self.depth_aoa):
            # MultiHeadAttention
            attn_output, _ = self.attn(q, k, v)
            
            # Concatenation of attn_output and q
            concat_v_with_q = torch.cat((attn_output, q), dim=-1)

            # Separate linear projections for the concatenated output
            projected_for_sigmoid = self.proj1(concat_v_with_q)
            projected_for_mult = self.proj2(concat_v_with_q)

            # Apply sigmoid and element-wise multiplication
            sigmoid_output = self.sigmoid(projected_for_sigmoid)
            mult_output = sigmoid_output * projected_for_mult

        # Apply residual connection and normalization
        q = self.norm(mult_output + x)

        # Feedforward layer and final residual connection and normalization
        out = self.ff(q)
        out = self.norm(out + q)

        return out



class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
        depth_aoa=None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        self.depth_aoa = depth * 3

        for _ in range(depth):
            self.layers.append(
                AoA(
                    dim,
                    heads,
                    dim_head,
                    dropout,
                    depth_aoa=self.depth_aoa,
                )
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


# classes


class AoATransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, ff_mult)

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)


x = torch.randint(0, 100, (1, 10))
model = AoATransformer(512, 1, 100)
out = model(x)
print(out.shape)
