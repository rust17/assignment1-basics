import torch
import torch.nn as nn

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # 通过“旋转”向量的方式，给模型注入位置信息
        # 在 __init__ 方法里，提前把所有可能用到的 sin 和 cos 值都算好。这样能大大提升计算效率，避免在每次前向传播时都重复计算三角函数
        # 公式：θ_i,k = i / (Θ^( (2k-2)/d ))
            # i 是词在序列中的位置 (position, from 0 to max_seq_len - 1)。
            # k 是特征维度的索引 (from 1 to d_k/2)。
            # Θ (theta) 是一个超参数，通常是 10000。
            # d 是 RoPE 作用的维度，这里是 d_k。
        # 先计算分母部分，torch.arange(0, d_k, 2) 会生成 [0, 2, 4, ..., d_k-2]，正好对应 2k-2，inv_freq 的形状是 (d_k/2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # 计算位置 i 和 inv_freq 的外积，就能得到所有 θ 值
        t = torch.arange(max_seq_len)

        # freqs 的形状就是 (max_seq_len, d_k/2)，这就是我们所有位置、所有维度对的旋转角度 θ 矩阵
        freqs = torch.outer(t, inv_freq)

        cos_values = torch.cos(freqs)
        sin_values = torch.sin(freqs)

        # self.register_buffer() 它告诉 PyTorch：“这个张量是模块状态的一部分（比如，应该跟着模型一起移动到 GPU），但它不是一个需要计算梯度的可学习参数。”
        self.register_buffer("cos_cached", cos_values, persistent=False)
        self.register_buffer("sin_cached", sin_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 用 token_positions 去预计算好的 cos_cached 和 sin_cached 里查表，拿到对应位置的旋转值，然后用这些值去“旋转” x

        # 查表
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 实现旋转
        # 把输入 x 从 (..., seq_len, d_k) 变成 (..., seq_len, d_k/2, 2)，也就是把 d_k 维两两一组
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        # 取出每一对 [x_even, x_odd]
        x1 = x_pairs[..., 0] # 所有偶数位的特征
        x2 = x_pairs[..., 1] # 所有奇数位的特征
        # 应用二维旋转公式
        # 就是 [[cos, -sin], [sin, cos]] @ [x1, x2] 的展开形式
        x_rot_1 = x1 * cos - x2 * sin
        x_rot_2 = x1 * sin + x2 * cos
        # 把旋转后的两部分重新组合起来
        x_roated_pairs = torch.stack([x_rot_1, x_rot_2], dim=-1)
        # 把形状还原回去
        return x_roated_pairs.flatten(start_dim=-2)
