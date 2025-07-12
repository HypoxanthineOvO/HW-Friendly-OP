import torch
from Modules.Approximate.ApproximateMatmul import Approx_Matmul

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