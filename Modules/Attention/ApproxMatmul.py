import torch

import torch

def Approx_Scale_Product_core(Q: torch.Tensor, K: torch.Tensor, max_iter: int) -> torch.Tensor:
    assert Q.dim() == 2 and K.dim() == 2, "Q and K must be 2D tensors"
    assert Q.size(1) == K.size(1), "Q and K must have the same feature dimension"
    Q_vec_size = Q.size(0)
    dtype, device = Q.dtype, Q.device
    
    total_score = []
    
    for q_id in range(Q_vec_size):
        q_mat = Q[q_id].repeat(K.size(0), 1)
        product = K * q_mat
        product_for_max = product.clone()
        product_for_min = product.clone()
        greedy_score = torch.zeros(K.size(0), dtype=dtype, device=device)
        for iter in range(max_iter):
            max_val = product_for_max.max()
            max_idx_flatten = product_for_max.argmax().item()
            max_idx_row, max_idx_col = divmod(max_idx_flatten, product_for_max.size(1))

            min_val = product_for_min.min()
            min_idx_flatten = product_for_min.argmin().item()
            min_idx_row, min_idx_col = divmod(min_idx_flatten, product_for_min.size(1))
            
            # Update the greedy score
            greedy_score[max_idx_row] += max_val
            product_for_max[max_idx_row, max_idx_col] = float('-inf')  # Set the max value to -inf to avoid selecting it again
            greedy_score[min_idx_row] += min_val
            product_for_min[min_idx_row, min_idx_col] = float('inf')  # Set the min value to inf to avoid selecting it again

        # Set the <0 values to 0
        greedy_score[greedy_score < 0] = 0
        total_score.append(greedy_score)
    total_score = torch.stack(total_score)
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