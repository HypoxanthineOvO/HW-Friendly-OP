import torch
import torch.nn as nn

import VSP


if __name__ == "__main__":
    torch.manual_seed(42)
    ########## 1. Linear ##########
    ## 1.1. Input Shape [N, IN_FEATURES]
    N = 256
    IN_FEATURES = 356
    OUT_FEATURES = 333
    BIAS = True
    

    linear_torch = nn.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        device = "cuda"
    )
    linear_A3 = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "A3",
            "Max_Iter": 1.0,  # 1.0 means 100% of total multiplications
            "Debug": False
        },
        device = "cuda"
    )
    
    
    linear_Row = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "Row",
            "Max_Iter": 1.0,  
            "Debug": False
        },
        device = "cuda"
    )
    
    print(f"Input Shape: [{N}, {IN_FEATURES}]")
    x = torch.randn(N, IN_FEATURES).to("cuda")
    VSP.printNamedTensor("Input Tensor", x.flatten()[:10])
    VSP.printNamedTensor("Weight", linear_A3.weight.flatten()[:10])
    VSP.printNamedTensor("Bias", linear_A3.bias.flatten()[:10])
    mem_start = VSP.getMemoryAllocated()
    output_torch = linear_torch(x)
    mem_after_torch = VSP.getMemoryAllocated()
    print(f"Memory after Torch Linear: {mem_after_torch} MB")
    print(f"Memory Usage: {mem_after_torch - mem_start:.2f} MB")
    
    linear_A3.load_torch_state_dict(linear_torch.state_dict())
    mem_after_load = VSP.getMemoryAllocated()
    output_A3 = linear_A3(x)
    mem_after_A3 = VSP.getMemoryAllocated()
    print(f"Memory after A3 Approx Linear: {mem_after_A3} MB")
    print(f"Memory Usage: {mem_after_A3 - mem_after_load:.2f} MB")
    
    VSP.printNamedTensor("Linear Output Torch", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine", output_A3.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_A3)
    print(f"Error: {err.item():.4f}")
    VSP.check( torch.allclose(output_torch, output_A3, atol = 1e-5), "Output Check")

    
    
    linear_Row.load_torch_state_dict(linear_torch.state_dict())
    mem_after_load = VSP.getMemoryAllocated()
    output_Row = linear_Row(x)
    mem_after_Row = VSP.getMemoryAllocated()
    print(f"Memory after Row Approx Linear: {mem_after_Row} MB")
    print(f"Memory Usage: {mem_after_Row - mem_after_load:.2f} MB")
    VSP.printNamedTensor("Linear Output Torch", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine", output_Row.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_Row)
    print(f"Error: {err.item():.4f}")
    VSP.check(torch.allclose(output_torch, output_Row, atol = 1e-5), "Output Check")
    exit()
    
    
    ## 1.2. More Input Shape
    N = (1,3,4)
    x = torch.randn(N + (IN_FEATURES,))
    print(f"Input Shape: {N + (IN_FEATURES,)}")
    VSP.printNamedTensor("Input Tensor", x.flatten()[:10])
    VSP.printNamedTensor("Weight", linear_A3.weight.flatten()[:10])
    VSP.printNamedTensor("Bias", linear_A3.bias.flatten()[:10])
    output_torch = linear_torch(x)
    output_mine = linear_A3(x)
    print(f"Output Shape: {output_torch.shape}, {output_mine.shape}")
    VSP.printNamedTensor("Linear Output Torch", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine", output_mine.flatten()[:10])
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