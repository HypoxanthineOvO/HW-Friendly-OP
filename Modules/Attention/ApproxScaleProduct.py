import torch

import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
while True:
    if os.path.basename(current_dir) == "HW-Friendly-OP":
        base_dir = current_dir
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise RuntimeError("HW-Friendly-OP directory not found.")
    current_dir = parent_dir
print(f"Base directory found: {base_dir}")
sys.path.append(os.path.join(base_dir, "ApproxMatmul"))
from ApproxMatmul.VSP import Approx_Matmul

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