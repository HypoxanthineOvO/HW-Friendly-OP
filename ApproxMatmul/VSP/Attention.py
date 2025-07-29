import torch
from .ApproximateMatmul import Approx_Matmul


def Approx_Scale_Product_core(Q: torch.Tensor, K: torch.Tensor, max_iter: int) -> torch.Tensor:
    total_score = Approx_Matmul(Q, K, max_iter)

    d_k = K.size(-1)  # Dimension of keys (D)
    total_score =  total_score/ (d_k ** 0.5)
    
    results = torch.nn.functional.softmax(total_score, dim=-1)
    
    return results

def Approx_Scale_Product(Q: torch.Tensor, K: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    assert Q.dim() == 3 and K.dim() == 3, "Q and K must be 3D tensors"
    assert Q.size(2) == K.size(2), "Q and K must have the same feature dimension"
    
    # Run in batch
    results = []
    for i in range(Q.size(0)):
        result = Approx_Scale_Product_core(Q[i], K[i], max_iter)
        results.append(result)
    results = torch.stack(results)
    return results


def Approx_QKV_Attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    attention_weights = Approx_Scale_Product(Q, K, max_iter=max_iter)
    output = torch.matmul(attention_weights, V)
    return attention_weights, output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
        embed_dim: int, # Dimension of the input embeddings
        num_heads: int, # Number of attention heads
        bias: bool = False, # Whether to use bias in linear layers
        approx: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads
        
        self.bias = bias
        
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
    
        if (approx):
            self.QKV_Attention = Approx_QKV_Attention
    
    def get_standard_statc_dict(self):
        standard_state_dict = {}
        # in_proj_weight: Concatenated weights for query, key, and value projections
        standard_state_dict['in_proj_weight'] = torch.cat([
            self.query_proj.weight,
            self.key_proj.weight,
            self.value_proj.weight
        ], dim=0)
        if (self.bias):
            # in_proj_bias: Concatenated biases for query, key, and value projections
            standard_state_dict['in_proj_bias'] = torch.cat([
                self.query_proj.bias,
                self.key_proj.bias,
                self.value_proj.bias
            ], dim=0)
        # out_proj_weight
        standard_state_dict['out_proj.weight'] = self.out_proj.weight
        if (self.bias):
            # out_proj_bias
            standard_state_dict['out_proj.bias'] = self.out_proj.bias
        return standard_state_dict
        
        
    def forward(self, x: torch.Tensor):
        # Dimension Check
        if x.dim() != 3:
            raise ValueError("Input tensor must be 3-dimensional (batch_size, seq_len, embed_dim), got {x.dim()} dimensions.")
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim != self.embed_dim:
            raise ValueError(f"Input embedding dimension ({embed_dim}) does not match the model's embedding dimension ({self.embed_dim}).")
        # Input projection
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Compute attention
        attn_weight, attn_output = self.QKV_Attention(q, k, v)
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output