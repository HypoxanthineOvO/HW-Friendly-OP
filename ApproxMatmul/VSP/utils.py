import torch

def printTensor(t: torch.Tensor):
    shape = t.shape
    if len(shape) == 1:
        for i in range(shape[0]):
            print(f"{t[i]:.4f}", end=" ")
        print()
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                print(f"{t[i, j]:.4f}", end=" ")
            print()
    else:
        t_for_print = t.view(-1, shape[-2], shape[-1])
        for i in range(t_for_print.shape[0]):
            print(f"Tensor {i}:")
            for j in range(t_for_print.shape[1]):
                for k in range(t_for_print.shape[2]):
                    print(f"{t_for_print[i, j, k]:.4f}", end=" ")
                print()
            print()
def printNamedTensor(name: str, t: torch.Tensor):
    print("=" * 50)
    print(f"{name}:")
    printTensor(t)

def check(check_result: bool, message: str):
    if not check_result:
        raise ValueError(f"Check failed: {message}")
    else:
        print(f"Check passed: {message}")