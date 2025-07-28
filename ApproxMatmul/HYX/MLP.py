import torch
import math
from .ApproximateMatmul import Approx_Matmul

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_iter: int|float = -1):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        if (isinstance(max_iter, int)):
            if (max_iter > 0):
                self.max_iter = max_iter
            else:
                self.max_iter = in_features * out_features // 2
        elif (isinstance(max_iter, float)):
            if (max_iter <= 0 or max_iter > 1):
                raise ValueError("max_iter must be an integer or a float between 0 and 1")
            else:
                self.max_iter = int(in_features * out_features // 2 * max_iter)
        else:
            raise TypeError("max_iter must be an integer or a float")
        print("Max Iterations:", self.max_iter)
        

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def load_torch_state_dict(self, state_dict):
        print(self.weight.shape)
        self.weight.data = state_dict['weight']
        print("Weight loaded:", self.weight.shape)
        if self.bias is not None:
            self.bias.data = state_dict['bias']

    def forward(self, x: torch.Tensor):
        mid = Approx_Matmul(x, self.weight, max_iter=self.max_iter)
        if self.bias is not None:
            mid += self.bias
        return mid