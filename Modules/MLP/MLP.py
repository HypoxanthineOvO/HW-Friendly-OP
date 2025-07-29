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

def MLP_Layer(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    activation: torch.nn.Module = None
):
    output = torch.matmul(weight.t(), x.t()).t()  # Transpose for correct shape
    if bias is not None:
        output += bias
    if activation is not None:
        output = activation(output)
    return output

def Approx_MLP_Layer(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    activation: torch.nn.Module = None,
    max_iter: int = 10
):
    output = Approx_Matmul(x, weight.t(), max_iter)
    if bias is not None:
        output += bias
    if activation is not None:
        output = activation(output)
    return output