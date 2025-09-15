import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        # 实现公式：RMSNorm(a_i) = (a_i / RMS(a)) * g_i
        # 其中，RMS(a) = sqrt( (1/d_model) * Σ(a_i^2) + ε )
        # 翻译一下：
        # 假设我们有一个输入向量 a (长度是 d_model)
        # 计算平方和：把向量 a 里的每一个元素都平方，然后加起来 (Σ(a_i^2))
        # 计算均值：把这个总和除以向量的维度 d_model
        # 加上 epsilon：给这个均值加上一个非常小的数 ε (比如 1e-5)。这是为了防止万一均值是 0，我们开方后除以 0 会出错
        # 开方：对上一步的结果开平方根。现在我们就得到了 RMS(a) 这个值
        # 归一化：把原始向量 a 的每一个元素都除以我们刚算出来的 RMS(a) 值
        # 应用增益：我们还需要一个模型可学习的参数 g (一个长度也是 d_model 的向量，初始化为全 1)。把上一步归一化后的向量，和这个增益向量 g 进行逐元素相乘

        self.eps = eps

        # 对应：模型可学习的增益参数 g
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # 为了防止计算 x ** 2 发生数值溢出（特别是用半精度 float16 训练时），需要先把输入 x 的数据类型提升至 float32，计算完后再转换回原来的类型
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # 实现公式：RMS(a) = sqrt( (1/d_model) * Σ(a_i^2) + ε )
        # 翻译成伪代码：RMS(a) = sqrt( mean(a^2) + ε )
        # a^2 在 PyTorch 里就是 x ** 2 或者 x.pow(2)
        # mean(·) 在 PyTorch 里就是 torch.mean(·, dim=..., keepdim=True)。这里的 dim=-1 表示沿着最后一个维度计算，keepdim=True 是个好习惯，它能让输出的维度和输入匹配，方便后续广播计算
        # sqrt(·) 就是 torch.sqrt(·)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 对应: 归一化
        x_normalized = x / rms

        # 对应: 应用增益
        output = x_normalized * self.weights

        return output.to(input_dtype)
