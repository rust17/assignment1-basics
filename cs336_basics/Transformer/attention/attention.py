import torch
from torch import Tensor
from jaxtyping import Bool
from ..activations.softmax import Softmax

class Attention:
    @classmethod
    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Bool) -> Tensor:
        # 实现 Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V

        # 计算原始分数 (Matmul 1)
        # transpose(-2, -1) 是为了确保我们交换的是最后两个维度（矩阵的行和列），而不会影响前面的批处理维度（比如 batch_size 和 num_heads）
        scores = q @ k.transpose(-2, -1)

        # 缩放 (Scaling)
        d_k = q.shape[-1]
        scaled_scores = scores / (d_k ** 0.5)

        # 应用掩码 (Masking)
        # mask 是一个布尔张量，False 的地方代表需要被遮盖
        # 我们需要把 scaled_scores 中对应 mask 为 False 的位置，都设置成一个非常小的负数（负无穷）
        torch.finfo(scaled_scores.dtype).min
        scaled_scores = scaled_scores.masked_fill(mask == False, torch.finfo(scaled_scores.dtype).min)

        # 计算注意力权重 (Softmax)，在最后一个维度（也就是 key 的序列长度维度）上进行归一化
        attention_weights = Softmax.softmax(scaled_scores, dim=-1)

        # 加权求和 (Matmul 2)
        return attention_weights @ v
