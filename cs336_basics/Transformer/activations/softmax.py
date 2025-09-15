import torch
from torch import Tensor

class Softmax:
    @classmethod
    def softmax(cls, x: Tensor, dim: int) -> Tensor:
        # 实现公式：softmax(v)_i = exp(v_i) / Σ(exp(v_j))

        # 找到最大值：沿着指定的 dim，找到 x 的最大值。torch.max(x, dim=dim, keepdim=True) 可以做到。keepdim=True 同样是为了方便后续的广播
        # torch.max 函数会返回两个值：一个是最大值本身 (values)，另一个是最大值对应的索引 (indices)
        # .values，这样我们就只取了最大值，忽略了索引
        max_val = torch.max(x, dim=dim, keepdim=True).values

        # 减去最大值：从 x 中减去我们刚找到的最大值
        # 计算指数：对上一步的结果，计算 exp，也就是 torch.exp()
        exps = torch.exp(x - max_val)

        # 计算分母：沿着 dim，对上一步的指数结果求和。torch.sum(..., dim=dim, keepdim=True)
        sum_exps = torch.sum(exps, dim=dim, keepdim=True)

        # 计算最终结果：用第 3 步的指数结果，除以第 4 步的分母
        return exps / sum_exps
