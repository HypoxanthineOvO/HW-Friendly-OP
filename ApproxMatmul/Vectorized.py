import torch
import torch.nn as nn

import VSP


if __name__ == "__main__":
    torch.manual_seed(42)
    ########## 1. Linear ##########
    ## 1.1. Input Shape [N, IN_FEATURES]
    N = 65536
    IN_FEATURES = 16
    OUT_FEATURES = 16
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
            "Max_Iter": 0.8,  # 1.0 means 100% of total multiplications
            "Debug": False
        },
        device = "cuda"
    )
    
    linear_Row = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        Approx_Config = {
            "Method": "Row",
            "Max_Iter": 0.8,  
            "Debug": False
        },
        device = "cuda"
    )
    
    #print(f"Input Shape: [{N}, {IN_FEATURES}]")
    x = torch.randn(N, IN_FEATURES).to("cuda")
    #VSP.printNamedTensor("Input Tensor:", x)
    #VSP.printNamedTensor("Weight: ", linear_A3.weight)
    #VSP.printNamedTensor("Bias: ", linear_A3.bias)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output_torch = linear_torch(x)
    end.record()
    torch.cuda.synchronize()  # Wait for the event to be recorded
    time_torch = start.elapsed_time(end)  # Time in milliseconds
    
    linear_A3.load_torch_state_dict(linear_torch.state_dict())
    start.record()
    output_A3 = linear_A3(x)
    end.record()
    torch.cuda.synchronize()  # Wait for the event to be recorded
    time_A3 = start.elapsed_time(end)  # Time in milliseconds
    #VSP.printNamedTensor("Linear Output Torch", output_torch.flatten()[:10])
    #VSP.printNamedTensor("Linear Output Mine", output_A3.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_A3)
    print(f"Error: {err.item():.4f}")
    VSP.check( torch.allclose(output_torch, output_A3), "Output Check")
    
    
    linear_Row.load_torch_state_dict(linear_torch.state_dict())
    start.record()
    output_Row = linear_Row(x)
    end.record()
    torch.cuda.synchronize()  # Wait for the event to be recorded
    time_Row = start.elapsed_time(end)  # Time in milliseconds
    VSP.printNamedTensor("Linear Output Torch", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine", output_Row.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_Row)
    print(f"Error: {err.item():.4f}")
    VSP.check(torch.allclose(output_torch, output_Row), "Output Check")
    
    print(f"Time Torch: {time_torch:.2f} ms")
    print(f"Time A3: {time_A3:.2f} ms")
    print(f"Time Row: {time_Row:.2f} ms")