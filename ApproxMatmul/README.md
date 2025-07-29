# Approximate Matmul VSP

This is the VSP Lab's Approximate Matmul package.

## How To Run?
在 ApproxMatmul 目录下运行
```bash
python Test.py
```

即可。

`Test.py` 的第一个测试是 `VSP.Linear`，几个超参数为：
- 第一行的 `torch.manual_seed()` 是固定的随机数种子，需要换随机数可以制定修改这里。
- `N`：输入向量有多少组
- `IN_FEATURES`：输入向量的维度
- `OUT_FEATURES`：输出维度


如果需要 Debug 信息，可以在 `VSP/MLP.py` 中的 `Linear - forward` 函数的 Approx_Matmul 函数参数后面加一个 `debug = True`
