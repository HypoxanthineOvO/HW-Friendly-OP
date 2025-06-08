import torch
from Modules.Attention.ApproxMatmul import Approx_Scale_Product

def Scaled_Product(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # Compute QK^T
    QKT = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale QK^T
    d_k = K.size(-1)
    QKT_scaled = QKT / (d_k ** 0.5)
    
    # Apply softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(QKT_scaled, dim=-1)
    return attention_weights

def QKV_Attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # TODO: Dimention Check

    # Compute attention weights
    attention_weights = Scaled_Product(Q, K)
    
    # Compute the output as a weighted sum of V
    output = torch.matmul(attention_weights, V)
    return attention_weights, output

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
        else:
            self.QKV_Attention = QKV_Attention
    
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
        print(f"Input shape: {x.shape}")
        # Input projection
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Compute attention
        print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        attn_weight, attn_output = self.QKV_Attention(q, k, v)
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output