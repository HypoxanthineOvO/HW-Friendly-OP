import torch
import torch.nn as nn

import VSP

import time
from functools import wraps

def timeit(description="Function"):
    """
    精确计时装饰器，专门处理 PyTorch GPU 操作的同步问题。
    
    Args:
        description (str): 函数描述，用于输出显示
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 开始前同步 GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            result = f(*args, **kwargs)
            
            # 结束后同步 GPU，确保所有 Kernel 完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            print(f"[{description}] 耗时: {elapsed_time:.4f} 秒")
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    torch.manual_seed(42)
    ########## 1. Linear ##########
    ## 1.1. Input Shape [N, IN_FEATURES]
    N = 1
    IN_FEATURES = 64
    OUT_FEATURES = 128
    BIAS = True
    

    linear_torch = nn.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS
    )
    linear_A3 = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "A3",
            "Max_Iter": 1.0,  # 1.0 means 50% of total multiplications
            "Debug": False
        }
    )
    
    linear_Row = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "Row",
            "Max_Iter": 1.0,  # 1.0 means 50% of total multiplications
            "Debug": False  
        }
    )
    
    linear_Block = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "Block",
            "Max_Iter": 1.0,  # 1.0 means 50% of total multiplications
            "Block_Size": 2,  
            "Debug": False
        }
    )
    
    print(f"Input Shape: [{N}, {IN_FEATURES}]")
    x = torch.randn(N, IN_FEATURES)
    VSP.printNamedTensor("Input Tensor:", x)
    VSP.printNamedTensor("Weight: ", linear_A3.weight)
    VSP.printNamedTensor("Bias: ", linear_A3.bias)
    time_start = time.time()
    output_torch = linear_torch(x)
    time_end = time.time()
    time_torch = time_end - time_start
    print(f"Torch Linear Time: {time_end - time_start:.4f} seconds")
    
    
    linear_A3.load_torch_state_dict(linear_torch.state_dict())
    time_start = time.time()
    output_A3 = linear_A3(x)
    time_end = time.time()
    time_A3 = time_end - time_start
    print(f"A3 Linear Time: {time_end - time_start:.4f} seconds")
    VSP.printNamedTensor("Linear Output Torch:", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine:", output_A3.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_A3)
    print(f"Error: {err.item():.4f}")
    VSP.check( torch.allclose(output_torch, output_A3), "Output Check")
    
    
    linear_Row.load_torch_state_dict(linear_torch.state_dict())
    time_start = time.time()
    output_Row = linear_Row(x)
    time_end = time.time()
    time_Row = time_end - time_start
    print(f"Row Linear Time: {time_end - time_start:.4f} seconds")
    VSP.printNamedTensor("Linear Output Torch:", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine(ROW):", output_Row.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_Row)
    print(f"Error: {err.item():.4f}")
    VSP.check(torch.allclose(output_torch, output_Row), "Output Check")
    
    linear_Block.load_torch_state_dict(linear_torch.state_dict())
    time_start = time.time()
    output_Block = linear_Block(x)
    time_end = time.time()
    time_Block = time_end - time_start
    print(f"Block Linear Time: {time_end - time_start:.4f} seconds")
    VSP.printNamedTensor("Linear Output Torch:", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine(BLOCK):", output_Block.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_Block)
    print(f"Error: {err.item():.4f}")
    VSP.check(torch.allclose(output_torch, output_Block), "Output Check")
    
    
    print("Time Summary:")
    print(f"Torch Linear Time: {time_torch:.4f} seconds")
    print(f"A3 Linear Time: {time_A3:.4f} seconds")
    print(f"Row Linear Time: {time_Row:.4f} seconds")
    print(f"Block Linear Time: {time_Block:.4f} seconds")
    
    exit()
    
    
    ## 1.2. More Input Shape
    N = (1,3,4)
    x = torch.randn(N + (IN_FEATURES,))
    print(f"Input Shape: {N + (IN_FEATURES,)}")
    VSP.printNamedTensor("Input Tensor:", x)
    VSP.printNamedTensor("Weight: ", linear_A3.weight)
    VSP.printNamedTensor("Bias: ", linear_A3.bias)
    output_torch = linear_torch(x)
    output_mine = linear_A3(x)
    print(f"Output Shape: {output_torch.shape}, {output_mine.shape}")
    VSP.printNamedTensor("Linear Output Torch:", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine:", output_mine.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_mine)
    print(f"Error: {err.item():.4f}")
    VSP.check( torch.allclose(output_torch, output_mine), "Same Output")
    
    exit()
    ########## 2. Attention ##########
    BATCH_SIZE = 2
    SEQ_LEN = 4
    EMBED_DIM = 8
    NUM_HEADS = 2
    approx = True
    
    attention_torch = nn.MultiheadAttention(
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, batch_first=True
    )
    
    