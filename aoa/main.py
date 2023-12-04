import torch 
from torch import nn
from zeta.nn import FeedForward, Attend


def exists(val):
    return val is not None



class AoA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        ff_mult: int = 4,
        depth: int = 1
    ):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.scale = dim_head ** -0.5
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
        
        self.attn_1 = Attend(
            dropout = dropout,
            causal=False,
            heads = heads
        )
        
        #Sigmoid
        self.sigmoid = nn.Sigmoid()
        
        # Feedforward
        self.ff = FeedForward(
            dim,
            dim,
            4,
        )
        
        
    def forward(self, x):
        device = x.device
        # Linear projection from x -> k, v, q
        k, v, q = self.k_proj(x), self.v_proj(x), self.q_proj(x)
        
        # MultiHeadAttention
        attn = self.attn(q, k, v)[0]
        print(attn.shape)
        # Unfurl attn because it's a tuple
        # attn = attn[0]
        # attn = self.attn_1(q, k, v)
        
        concat_v_with_q = torch.concat((attn, q))
        
        # Proj concat_v_with_q
        projected_concat_for_sigmoid, projected_concat_for_mult = self.proj(concat_v_with_q)
        
        # Sigmoid projected_concat_for_sigmoid
        sigmoided_concat_linear = self.sigmoid(projected_concat_for_sigmoid)
        
        # Mult sigmoided_concat_linear with projected_concat_for_mult
        mult_sigmoid_with_linear_concat = torch.matmul(sigmoided_concat_linear, projected_concat_for_mult)
        
        # layernorm and add x
        normed_mult_sigmoid = self.norm(mult_sigmoid_with_linear_concat) + x
        
        # feedforward
        ffn_normed_mult_sigmoid = self.ff(normed_mult_sigmoid) 
        
        # Add and norm
        out = self.norm(ffn_normed_mult_sigmoid) + normed_mult_sigmoid
        
        return out
    

x = torch.randn(1, 10, 512)
aoa = AoA(512, 8, 64, 0.1)
out = aoa(x)
print(out.shape)
