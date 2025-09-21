import torch
import math
from .ApproximateMatmul import Approx_Matmul

class Linear(torch.nn.Module):
    def __init__(self, 
                 in_features: int, out_features: int, bias: bool = True, 
                 device: str = "cpu",
                 Approx_Config: dict = {
                     "Method": "A3",
                     "Max_Iter": 1.0,
                     "Debug": False
                 },
                 ):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self.to(device)
        
        # Parse Approx_Config
        method = Approx_Config.get("Method", "A3")
        if (method == "A3" or method == "Row"):
            max_iter = Approx_Config.get("Max_Iter", 1.0)
            debug = Approx_Config.get("Debug", False)
            Final_Approx_Config = {
                "Method": method,
                "Max_Iter": max_iter,
                "Debug": debug
            }
            # Check Validity
            if not (isinstance(max_iter, int) or isinstance(max_iter, float)):
                raise TypeError("max_iter must be an integer or a float")
            if isinstance(max_iter, int):
                if max_iter < 0:
                    raise ValueError("max_iter must be non-negative")
            if isinstance(max_iter, float):
                if max_iter <= 0 or max_iter > 1:
                    raise ValueError("max_iter as float must be in (0, 1]")
                
                if (method == "A3"):
                    int_iter = int(in_features * out_features // 2 * max_iter)
                    print(f"Approximate Iterations: {int_iter} for A3 Method (Feature Size: {in_features}, Output Size: {out_features})")
                elif (method == "Row"):
                    int_iter = int(in_features / 2 * max_iter)
                    print(f"Approximate Iterations: {int_iter} for Row Method (Feature Size: {in_features})")
                Final_Approx_Config["Max_Iter"] = int_iter
            self.Approx_Config = Final_Approx_Config
        else:
            raise ValueError(f"Unknown Approximation Method: {method}")
        self.debug = self.Approx_Config["Debug"]
        
        self.chunk_size = 512*512

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def load_torch_state_dict(self, state_dict):
        if self.debug:
            print("Loading state dict...")
            print("Weight shape:", state_dict['weight'].shape)
            if self.bias is not None:
                print("Bias shape:", state_dict['bias'].shape)
        self.weight.data = state_dict['weight']
        
        if self.bias is not None:
            self.bias.data = state_dict['bias']

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        output_shape = x_shape[:-1] + (self.weight.shape[0],)
        x = x.view(-1, x_shape[-1])  # Flatten the input tensor

        x.to("cpu")
        # Chunk the input tensor if it is too large
        if (x.numel() > self.chunk_size):
            x_chunks = x.split(self.chunk_size, dim=0)
            output_chunks = []
            for chunk in x_chunks:
                chunk = chunk.to(self.weight.device)
                output = Approx_Matmul(
                    chunk, self.weight, Approx_Config=self.Approx_Config
                ).cpu()
                output_chunks.append(output)
                
            mid = torch.cat(output_chunks, dim=0).to(x.device)
        else:
            x = x.to(self.weight.device)

            mid = Approx_Matmul(
                x, self.weight, Approx_Config=self.Approx_Config
                )
        if self.bias is not None:
            mid += self.bias
        
        result = mid.view(*output_shape)
        return result