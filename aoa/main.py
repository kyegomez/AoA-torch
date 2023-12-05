"""
Attention on Attention Implementation
This is a practice implementation after randomly finding it on Lucidrain's repo, I'm implementing the model architecture just for practice!

Basically the architecture is: x => q, k, v -> multihead attn with residual q -> concat -> 2 linear projects ->sigmoid -> mult -> add -> norm -> ffn -> add -> norm with residual of first add and norm

"""

import torch
from torch import nn
from zeta.nn import FeedForward, Attend


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

    

x = torch.randn(1, 10, 512)
model = AoA(512, 8, 64, 0.1)
out = model(x)
print(out.shape)
