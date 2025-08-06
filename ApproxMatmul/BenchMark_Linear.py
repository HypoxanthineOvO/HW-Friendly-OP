import torch
import torch.nn as nn

import VSP

import pandas as pd
import matplotlib.pyplot as plt

def computeLoss(
    N: int, IN_FEATURES: int, OUT_FEATURES: int,
    Approx_Config: dict
):
    linear_torch = nn.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = True
    )
    linear_approx = VSP.Linear(
        in_features = IN_FEATURES, out_features = OUT_FEATURES,
        bias = True,
        Approx_Config = Approx_Config
    )
    
    x = torch.randn(N, IN_FEATURES)
    output_torch = linear_torch(x)
    
    linear_approx.load_torch_state_dict(linear_torch.state_dict())
    output_approx = linear_approx(x)
    
    err = torch.nn.functional.mse_loss(output_torch, output_approx)
    related_err = err / torch.linalg.norm(output_torch)
    return (err.item(), related_err.item()), output_torch, output_approx


if __name__ == "__main__":
    Ns = [128]
    IN_FEATURES = [16, 32]
    OUT_FEATURES = [16, 32]
    Approx_rate = [0.2, 0.5, 0.8, 0.9, 0.95]
    Methods = ["A3", "Row"]
    
    Loss_Records = {}
    summary_rows = []
    for method in Methods:
        Loss_Records[method] = {}
        for N in Ns:
            Loss_Records[method][N] = {}
            for in_f in IN_FEATURES:
                Loss_Records[method][N][in_f] = {}
                for out_f in OUT_FEATURES:
                    Loss_Records[method][N][in_f][out_f] = {}
                    for rate in Approx_rate:
                        Approx_Config = {
                            "Method": method,
                            "Max_Iter": rate,
                            "Debug": False
                        }
                        err, out_torch, out_approx = computeLoss(
                            N, in_f, out_f, Approx_Config
                        )
                        Loss_Records[method][N][in_f][out_f][rate] = err
                        summary_rows.append({
                            "Method": method,
                            "N": N,
                            "IN_FEATURES": in_f,
                            "OUT_FEATURES": out_f,
                            "Approx_rate": rate,
                            "MSE_Loss": err
                        })
                        print(f"Method: {method}, N: {N}, IN: {in_f}, OUT: {out_f}, Rate: {rate:.2f} => MSE Loss: {err[1]:.6f}")
    # Print Summary
    print("\n" + "=" * 80)
    print("Summary of MSE Loss:")
    for method in Methods:
        print(VSP.color(f"\nMethod: {method}", "green", "bold"))
        for N in Ns:
            for in_f in IN_FEATURES:
                for out_f in OUT_FEATURES:
                    print(f"N={N}, IN={in_f}, OUT={out_f}: |", end="")
                    for rate in Approx_rate:
                        errs = Loss_Records[method][N][in_f][out_f][rate]
                        err, related_err = errs
                        print(f" {rate:.2f}: {err:.2e} / {related_err: .2e} ", end=" |")
                    print()
    print("=" * 80)

    # Save summary to DataFrame and CSV
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("MSE_Loss_Summary.csv", index=False)
    
    # Visualization: plot all methods and N in one figure with subplots
    num_methods = len(Methods)
    num_Ns = len(Ns)
    fig, axes = plt.subplots(num_methods, num_Ns, figsize=(6 * num_Ns, 5 * num_methods), squeeze=False)

    y_min, y_max = 1e-16, 1e-2  # Set consistent y-axis limits

    for i, method in enumerate(Methods):
        for j, N in enumerate(Ns):
            ax = axes[i][j]
            for in_f in IN_FEATURES:
                for out_f in OUT_FEATURES:
                    rates = []
                    losses = []
                    for rate in Approx_rate:
                        errs = Loss_Records[method][N][in_f][out_f][rate]
                        _, related_err = errs
                        rates.append(rate)
                        losses.append(related_err)
                    ax.plot(rates, losses, marker='o', label=f"IN={in_f}, OUT={out_f}")
            ax.set_title(f"Method: {method}, N={N}")
            ax.set_xlabel("Approximation Rate")
            ax.set_ylabel("MSE Loss")
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
            ax.legend()
            ax.grid(True)
            ax.set_xticks(Approx_rate)
    plt.tight_layout()
    plt.savefig("MSE_Loss_AllMethods.png")
    plt.close()