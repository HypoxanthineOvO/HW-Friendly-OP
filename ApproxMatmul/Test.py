import torch
import torch.nn as nn

import VSP


if __name__ == "__main__":
    torch.manual_seed(42)
    ########## 1. Linear ##########
    N = 30
    IN_FEATURES = 10
    OUT_FEATURES = 5
    BIAS = True
    linear_torch = nn.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS
    )
    linear_mine = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = BIAS,
        max_iter = 0.8
    )
    linear_mine.load_torch_state_dict(linear_torch.state_dict())
    
    x = torch.randn(N, IN_FEATURES)
    VSP.printNamedTensor("Input Tensor:", x)
    VSP.printNamedTensor("Weight: ", linear_mine.weight)
    VSP.printNamedTensor("Bias: ", linear_mine.bias)
    output_torch = linear_torch(x)
    output_mine = linear_mine(x)
    VSP.printNamedTensor("Linear Output Torch:", output_torch.flatten()[:10])
    VSP.printNamedTensor("Linear Output Mine:", output_mine.flatten()[:10])
    err = torch.nn.functional.mse_loss(output_torch, output_mine)
    print(f"Error: {err.item():.4f}")
    VSP.check( torch.allclose(output_torch, output_mine), "Same Output")