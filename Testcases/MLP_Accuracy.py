import torch
import os, sys
import numpy as np

sys.path.append("..")

import Modules.MLP.MLP as MLP
import matplotlib.pyplot as plt


def test_approx_mlp_layer(
    N: int,
    D_in: int,
    D_out: int,
    max_iter_ratio: float = 0.8,
    std: float = 1.0,
    dtype: torch.dtype = torch.float32,
    activation: torch.nn.Module = torch.nn.ReLU(),
    seed: int = 42
):
    # 设置随机种子
    torch.manual_seed(seed)

    # 构造输入和参数
    mean = 0.0
    std = 1
    
    x = torch.normal(mean=mean, std=std, size=(N, D_in)).to(dtype=dtype)
    weight = torch.normal(mean=mean, std=std, size=(D_in, D_out)).to(dtype=dtype)
    bias = torch.normal(mean=mean, std=std, size=(D_out,)).to(dtype=dtype)

    # 正确输出
    output = MLP.MLP_Layer(x, weight, bias, activation)

    # 完整迭代次数
    complete_iter = weight.shape[0] * weight.shape[1] // 2
    max_iter = int(complete_iter * max_iter_ratio)

    # 近似输出
    approx_output = MLP.Approx_MLP_Layer(x, weight, bias, activation, max_iter)

    # 计算差异（L2 Loss）
    difference = torch.nn.functional.mse_loss(output, approx_output)
    normalized_difference = difference / torch.norm(output)
    # 打印信息
    print(f"Input size: ({N}, {D_in}), Weight size: ({D_in}, {D_out})")
    print(f"Complete iteration: {complete_iter}")
    print(f"Max iteration: {max_iter}")
    print(f"Difference between outputs (MSE): {difference:.6f} ({normalized_difference:.6f})")

    return difference, normalized_difference
if __name__ == "__main__":
    # 测试参数
    N = 10
    D_in_values = [4, 16, 64, 256]
    MAX_ITER_RATIOs = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95]
    D_out = 64
 
    activations = [
        torch.nn.ReLU(),
        torch.nn.LeakyReLU(),
        torch.nn.Sigmoid(),
        torch.nn.Tanh()
    ]
 
    # 存储每个激活函数对应的差异数据
    all_differences = []
 
    # 先跑完所有数据，获取最大 y 值用于统一设置 y 轴范围
    print("Running all tests to collect data...")
    for activation in activations:
        differences = []
        for D_in in D_in_values:
            for max_iter_ratio in MAX_ITER_RATIOs:
                print(f"Testing with D_in={D_in}, max_iter_ratio={max_iter_ratio}, {activation.__class__.__name__}")
                _, diff = test_approx_mlp_layer(N, D_in, D_out, max_iter_ratio, activation=activation)
                differences.append(diff.item())
 
        all_differences.append(np.array(differences).reshape(len(D_in_values), len(MAX_ITER_RATIOs)))
 
    # 找到所有差异中的最大值，用于统一 y 轴范围
    max_diff = np.max(all_differences)
 
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
 
    x = np.arange(len(D_in_values))
    width = 0.15
 
    for idx, (ax, activation_name, diff_data) in enumerate(zip(axes, [a.__class__.__name__ for a in activations], all_differences)):
        for i, ratio in enumerate(MAX_ITER_RATIOs):
            ax.bar(x + i * width, diff_data[:, i], width, label=f"ratio={ratio:.2f}")
 
        ax.set_title(f"{activation_name}", fontsize = 24)
        # ax.set_xlabel("Input Dimension (D_in)")
        #if idx % 2 == 0:
        #    ax.set_ylabel("Difference (MSE)")
        ax.set_xticks(x + width * (len(MAX_ITER_RATIOs) - 1) / 2)
        ax.set_xticklabels(D_in_values, fontsize = 20)
        # Set y ticks
        ax.grid(axis='y')
        ax.text(0.05, 0.95, f"d = {D_in_values}", transform=ax.transAxes, fontsize=16,
                verticalalignment='top')
 
    # 统一设置 y 轴范围
    plt.ylim(0, max_diff * 1.2)
    # Set y ticks
    for ax in axes:
        ax.set_yticks(np.linspace(0, max_diff, 5))
        ax.set_yticklabels([f"{y:.6f}" for y in np.linspace(0, max_diff, 5)], fontsize=14)

    # 添加图例（只在第一个子图上显示）
    axes[1].legend(title="max_iter_ratio", bbox_to_anchor=(1.05, 1), loc='upper left')
 
    plt.tight_layout()
    plt.savefig("Difference_vs_Input_Dimension_All_Activations.png")