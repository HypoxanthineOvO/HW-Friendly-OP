import torch
from .utils import printTensor, printNamedTensor, getMemoryAllocated
from tqdm import trange, tqdm

def Approx_Matmul(
    A: torch.Tensor, B: torch.Tensor,
    Approx_Config: dict = {
        "Method": "A3",
        "Max_Iter": 10,
        "Debug": False
    },
):
    method = Approx_Config.get("Method", "A3")
    debug = Approx_Config.get("Debug", False)
    
    if (method == "A3"):
        max_iter = Approx_Config.get("Max_Iter", 10)
        return Approx_Matmul_A3(A, B, max_iter=max_iter, debug=debug)
    elif (method == "Row"):
        max_iter = Approx_Config.get("Max_Iter", 10)
        return Approx_Matmul_Row(A, B, max_iter = max_iter, debug = debug)
    elif (method == "Block"):
        max_iter = Approx_Config.get("Max_Iter", 10)
        block_size = Approx_Config.get("Block_Size", 3)
        return Approx_Matmul_Block(A, B, max_iter=max_iter, block_size=block_size, debug=debug)
    else:
        raise ValueError(f"Unknown Approximation Method: {method}")

def Approx_Matmul_A3_naive(
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

def Approx_Matmul_A3(
    A: torch.Tensor, B: torch.Tensor, max_iter: int = 10,
    debug: bool = False
    ) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    assert A.size(1) == B.size(1), f"A({A.shape}) and B({B.shape}) must have the same feature dimension"
    A_vec_size = A.size(0)
    output_size = B.size(0)
    dtype, device = A.dtype, A.device
    
    total_score = torch.zeros((A_vec_size, B.size(0)), dtype=dtype, device = device, requires_grad = False)
    
    if debug:
        print("*" * 50)
    
    iterator = trange(A_vec_size) if debug else range(A_vec_size)
    for a_id in iterator:
        if debug:
            printNamedTensor(f"A[{a_id}]", A[a_id])
        
        a_mat = A[a_id].repeat(B.size(0), 1).to(device)
        product = (B * a_mat).detach() # Critical!!! If not detach, memory will keep increasing because of autograd
        
        product_flatten = product.flatten() # No Memory Increase
        max_k_values, max_k_indices = torch.topk(product_flatten, k=max_iter)
        min_k_values, min_k_indices = torch.topk(product_flatten, k=max_iter, largest=False)
        
        # Accumulate the scores with column val
        total_score[a_id].index_add_(
            0, 
            max_k_indices // product.size(1), 
            max_k_values
        )
        total_score[a_id].index_add_(
            0, 
            min_k_indices // product.size(1), 
            min_k_values
        )

        del a_mat, product, product_flatten, max_k_values, max_k_indices, min_k_values, min_k_indices
        torch.cuda.empty_cache()

    
    if debug:
        printNamedTensor("Total Score:", total_score)
        print("*" * 50)
    return total_score


def Approx_Matmul_Row(
    A: torch.Tensor, B: torch.Tensor,
    max_iter: int = 10, debug: bool = False
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
        
        row_sorted_matrix = torch.sort(product, dim=1, descending=True)[0]
        row_scores_max = row_sorted_matrix[:, :max_iter].sum(dim = 1)
        row_scores_min = row_sorted_matrix[:, -max_iter: ].sum(dim = 1)
        if debug:
            print(f"Max Iter: {max_iter}")
            printNamedTensor("Product ",product)
            printNamedTensor("Row Sorted Matrix:", row_sorted_matrix)
            printNamedTensor("Row Scores Max:", row_scores_max)
            printNamedTensor("Row Scores Min:", row_scores_min)
        row_scores = row_scores_max + row_scores_min
        total_score.append(row_scores)

    total_score = torch.stack(total_score)
    
    if debug:
        printNamedTensor("Total Score:", total_score)
        print("*" * 50)
    return total_score

"""
/**
 * @brief   将矩阵按块分组排序的近似矩阵乘法
 * @param   max_iter:每个行选择的最值数量,block_size:块的行数
 */
""" 
def Approx_Matmul_Block(
    A: torch.Tensor, B: torch.Tensor,
    max_iter: int = 10, block_size: int = 3, debug: bool = False
) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    assert A.size(1) == B.size(1), f"A({A.shape}) and B({B.shape}) must have the same feature dimension"
    A_vec_size = A.size(0)
    dtype, device = A.dtype, A.device
    
    total_score = []

    if debug:
        print("*" * 50)
        print(f"Block size: {block_size}, Max iter per row: {max_iter}")
    
    # 算出总block数，向上取整
    num_blocks = (B.size(0) + block_size - 1) // block_size

    for a_id in range(A_vec_size):
        if debug:
            printNamedTensor(f"A[{a_id}]", A[a_id])

        a_mat = A[a_id].repeat(B.size(0), 1)
        product = B * a_mat
        
        row_scores = torch.zeros(B.size(0), dtype=dtype, device=device)
        
        for block_id in range(num_blocks):
            start_row = block_id * block_size
            # 用于防止最后一个block大小不匹配
            end_row = min((block_id + 1) * block_size, B.size(0))
            actual_block_size = end_row - start_row
            
            block = product[start_row : end_row, :]  
            
            if debug:
                print(f"\nBlock {block_id}: processing rows {start_row} to {end_row-1}")
                printNamedTensor(f"Block {block_id} data", block)
            
            block_flatten = block.flatten()  
            
            num_max_values = actual_block_size * max_iter
            num_min_values = actual_block_size * max_iter
            
            if debug:
                print(f"Block has {total_elements} elements, selecting {num_max_values} max and {num_min_values} min values")
            
            max_values, max_indices = torch.topk(block_flatten, k=num_max_values, largest=True)

            min_values, min_indices = torch.topk(block_flatten, k=num_min_values, largest=False)
            
            if debug:
                print(f"Selected max values: {max_values[:5] if len(max_values) > 0 else 'None'}...")
                print(f"Selected min values: {min_values[:5] if len(min_values) > 0 else 'None'}...")

            feature_dim = block.size(1)
            for idx, value in zip(max_indices, max_values):
                # 在block里的索引0,1,2
                row_in_block = idx.item() // feature_dim
                # 在整个矩阵中的索引  
                global_row = start_row + row_in_block     
                row_scores[global_row] += value.item()     
            
            for idx, value in zip(min_indices, min_values):
                row_in_block = idx.item() // feature_dim
                global_row = start_row + row_in_block
                row_scores[global_row] += value.item()
            
            if debug:
                print(f"Row scores after processing block {block_id}:")
                for i in range(start_row, end_row):
                    print(f"  Row {i}: {row_scores[i].item():.4f}")
        
        total_score.append(row_scores)

    total_score = torch.stack(total_score)
    
    if debug:
        printNamedTensor("Total Score:", total_score)
        print("*" * 50)
    return total_score