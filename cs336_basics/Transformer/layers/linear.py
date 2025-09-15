import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 在 PyTorch 里，如果你想让一个张量 (tensor) 成为模型的可学习参数（也就是在训练中被优化的东西），你需要把它用 nn.Parameter() 包起来
        tensor_w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.W = nn.Parameter(tensor_w)

        # 实现公式：
        # Linear weights: N(µ = 0, σ^2 = 2 / (din + dout)) truncated at [−3σ, 3σ].
        # std 是标准差(也就是公式里的 σ)
        std = math.sqrt(2 / (in_features + out_features))
        # 函数名最后的下划线 _ 在 PyTorch 里是个约定，表示这个函数会原地修改 (in-place) 输入的张量，而不是返回一个新的
        # a 和 b 就是截断的范围 [-3σ, 3σ]
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在 Python 3.5 之后，专门为矩阵乘法引入了一个新的运算符：@ 符号
        # 所以，如果你有两个张量 A 和 B，它们的矩阵乘法就是 C = A @ B。这比用 torch.matmul(A, B) 要简洁得多
        # 数学上，我们习惯写 y = Wx，其中 x 是列向量
        # 在 PyTorch/NumPy 里，由于内存是行优先的，我们通常处理的是行向量。线性变换就变成了 y = xW^T（W^T 表示 W 的转置）
        # 我们的 self.W 形状是 (out_features, in_features)。为了让 x (..., in_features) 能和它相乘，我们需要把 self.W 转置一下，变成 (in_features, out_features)
        # 在 PyTorch 里，一个张量 A 的转置非常简单，就是 A.T
        # (..., in_features) @ (in_features, out_features) -> 结果形状是 (..., out_features)
        return x @ self.W.T
