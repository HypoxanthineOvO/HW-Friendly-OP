import torch
from .utils import printTensor, printNamedTensor

def Approx_Matmul(
    A: torch.Tensor, B: torch.Tensor, max_iter: int = 10,
    debug: bool = False
    ) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    assert A.size(1) == B.size(1), f"A({A.shape}) and B({B.shape}) must have the same feature dimension"
    A_vec_size = A.size(0)
    dtype, device = A.dtype, A.device
    
    total_score = []
    
    if debug:
        print("*" * 50)
    
    for a_id in range(A_vec_size):
        if debug:
            printNamedTensor(f"A[{a_id}]", A[a_id])
        a_mat = A[a_id].repeat(B.size(0), 1)
        product = B * a_mat
        product_for_max = product.clone()
        product_for_min = product.clone()
        greedy_score = torch.zeros(B.size(0), dtype=dtype, device=device)
        for iter in range(max_iter):
            max_val = product_for_max.max()
            max_idx_flatten = product_for_max.argmax().item()
            max_idx_row, max_idx_col = divmod(max_idx_flatten, product_for_max.size(1))

            min_val = product_for_min.min()
            min_idx_flatten = product_for_min.argmin().item()
            min_idx_row, min_idx_col = divmod(min_idx_flatten, product_for_min.size(1))
            
            if debug:
                print(f"Iteration {iter + 1}:")
                print(f"Max Value: {max_val:.4f} at ({max_idx_row}, {max_idx_col})")
                print(f"Min Value: {min_val:.4f} at ({min_idx_row}, {min_idx_col})")
            # Update the greedy score
            greedy_score[max_idx_row] += max_val
            product_for_max[max_idx_row, max_idx_col] = float('-inf')  # Set the max value to -inf to avoid selecting it again
            greedy_score[min_idx_row] += min_val
            product_for_min[min_idx_row, min_idx_col] = float('inf')  # Set the min value to inf to avoid selecting it again

        # Set the <0 values to 0
        total_score.append(greedy_score)
    total_score = torch.stack(total_score)
    
    if debug:
        printNamedTensor("Total Score:", total_score)
        print("*" * 50)
    return total_score
