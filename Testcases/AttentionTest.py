import torch
import os, sys

sys.path.append(".")

import Modules.Attention.Attention as Attention


if __name__ == "__main__":
    # Test case 1: Scaled Dot-Product
    Q = torch.rand(2, 256, 256)
    K = torch.rand(2, 256, 256)
    V = torch.rand(2, 256, 256)
    weights, output = Attention.QKV_Attention(Q, K, V)
    
    reference = torch.nn.functional.scaled_dot_product_attention(
        query=Q, key=K, value=V
    )
    if (torch.allclose(output, reference)):
        print("=" * 10, " Check Pass " , "=" * 10)
    else:
        print("Output:\n", output)
        print("Reference Output:\n", reference)
        loss = torch.nn.functional.mse_loss(output, reference)
        print(f"Error! Loss = {loss}")
    
    # Test case 2: Multi-Head Attention
    num_heads = 8
    embed_dim = 256
    mha_mine = Attention.MultiHeadAttention(embed_dim, num_heads, bias = True)
    
    x = torch.rand(4, 128, embed_dim)
    output = mha_mine(x)
    
    mha_ref = torch.nn.MultiheadAttention(
        embed_dim, num_heads,
        #bias = False
        ).eval()
    print(mha_ref.state_dict().keys())
    mha_ref.load_state_dict(mha_mine.get_standard_statc_dict())
    mha_ref_output, _ = mha_ref(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
    
    if (torch.allclose(output, mha_ref_output.transpose(0, 1), atol = 1e-4)):
        print("=" * 10, " Check Pass " , "=" * 10)
    else:
        print("Output:\n", output)
        print("Reference Output:\n", mha_ref_output.transpose(0, 1))
        loss = torch.nn.functional.mse_loss(output, mha_ref_output.transpose(0, 1))
        print(f"Error! Loss = {loss}")