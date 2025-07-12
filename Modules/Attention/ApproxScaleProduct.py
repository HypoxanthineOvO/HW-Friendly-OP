import torch
from Modules.Approximate.ApproximateMatmul import Approx_Matmul

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