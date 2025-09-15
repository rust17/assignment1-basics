import torch
import torch.nn as nn
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # 实现 SwiGLU 前馈网络
        # 公式：FFN(x) = W2(SiLU(W1x) ⊙ W3x)
        # 这里面包含了三个线性层（W1, W2, W3）和一个特殊的激活函数 SiLU，以及一个逐元素相乘操作 ⊙
        # 在 __init__ 方法里，我们需要把这三个线性层准备好。它们都是我们实现的 Linear 模块的实例
        # 根据讲义：where x ∈ R^dmodel, W1, W3 ∈ R^(dff × dmodel), W2 ∈ R^(dmodel × dff), and canonically, dff = 8/3 * dmodel
        # w1 的权重形状是 (d_ff, d_model)，w3 和 w1 类似
        # w2 的权重形状是 (d_model, d_ff)
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 公式：W2(SiLU(W1x) ⊙ W3x)
        # 分路计算
        x1 = self.w1(x)
        x3 = self.w3(x)

        # 实现 SiLU，SiLU 公式是 x * sigmoid(x)，可以直接使用 pytorch 函数
        silu_x1 = x1 * torch.sigmoid(x1)

        # 逐元素相乘（门控）：把 SiLU 的结果和 x3 相乘
        gated = silu_x1 * x3

        # 把门控结果送入 w2
        return self.w2(gated)
