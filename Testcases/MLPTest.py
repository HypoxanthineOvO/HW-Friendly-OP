import torch
import os, sys

sys.path.append("..")

import Modules.MLP.MLP as MLP

if __name__ == "__main__":
    # Test case 1: MLP Layer
    x = torch.rand(5, 16)
    weight = torch.rand(16, 32)
    bias = torch.rand(32)
    activation = torch.nn.ReLU()

    output = MLP.MLP_Layer(x, weight, bias, activation)
    
    # Reference output using standard matmul
    reference_mlp = torch.nn.Linear(16, 32, bias=True)
    reference_mlp.weight = torch.nn.Parameter(weight.t())
    reference_mlp.bias = torch.nn.Parameter(bias)
    reference_output = reference_mlp(x)
    reference_output = activation(reference_output)

    if torch.allclose(output, reference_output):
        print("=" * 10, " MLP Layer Check Pass ", "=" * 10)
    else:
        print(f"Output ({output.shape}):\n", output)
        print(f"Reference Output({reference_output.shape}):\n", reference_output)
        loss = torch.nn.functional.mse_loss(output, reference_output)
        print(f"Error! Loss = {loss}")
    
    # Test case 2: Approximate MLP Layer
    max_iter = 200
    approx_output = MLP.Approx_MLP_Layer(x, weight, bias, activation, max_iter)
    
    if torch.allclose(approx_output, reference_output, rtol=1e-2):
        print("=" * 10, " Approx MLP Layer Check Pass ", "=" * 10)
    else:
        print(f"Approx Output ({approx_output.shape}):\n", approx_output)
        print(f"Reference Output ({reference_output.shape}):\n", reference_output)
        loss = torch.nn.functional.mse_loss(approx_output, reference_output)
        print(f"Approx Error! Loss = {loss}")