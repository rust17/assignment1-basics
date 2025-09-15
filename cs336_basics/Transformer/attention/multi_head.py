import torch
import torch.nn as nn
from ..layers.linear import Linear
from .attention import Attention
from .rope import ROPE

class MultipleAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: ROPE | None = None):
        # 实现一个包含 WQ, WK, WV, WO 权重，并且支持多头和因果屏蔽的自注意力模块
        super().__init__()

        # 创建 4 个线性层，作为我们的权重矩阵
        self.wq = Linear(in_features=d_model, out_features=d_model)
        self.wk = Linear(in_features=d_model, out_features=d_model)
        self.wv = Linear(in_features=d_model, out_features=d_model)
        self.wo = Linear(in_features=d_model, out_features=d_model)

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 输入 x 的形状是 (batch_size, seq_len, d_model)
        # q, k, v 的形状都还是 (batch_size, seq_len, d_model)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 当你定义一个层，比如 Linear(in_features=10, out_features=20)，你只关心最后一个维度（特征维度）的变化。
        # 当你调用它时，你可以传入：
        # 形状为 (10,) 的单个向量
        # 形状为 (32, 10) 的一批向量 (batch_size=32)
        # 形状为 (32, 100, 10) 的一批序列 (batch_size=32, seq_len=100)
        # 模块的 forward 方法会自动处理这些额外的“批处理维度”（... in "... d_in"），并保持它们不变。这就是为什么我们在 transpose 和 view 操作中需要小心处理，确保只操作我们关心的 seq_len, num_heads, d_k 维度，而不破坏前面的 batch_size 等维度
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # 拆分成多头，需要把 d_model 这个维度拆成 num_heads 和 d_k
        # 同时，transpose(1, 2) 是为了方便后续矩阵乘法，我们需要把 num_heads 这个维度换到前面来，变成批处理维度的一部分
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 构建因果掩码
        # 需要一个 (seq_len, seq_len) 的上三角矩阵，来阻止未来的信息泄露
        # 下面这个代码可以生成这个矩阵，True 的地方代表需要被屏蔽
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # 期待的 mask 是 True 代表保留，所以需要对这个三角矩阵取反 ~
        causal_mask = ~mask

        # attention_output 的形状是 (batch_size, num_heads, seq_len, d_k)
        attention_output = Attention.scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 合并多头
        # 需要把 num_heads 和 d_k 再合并回 d_model
        # 首先，把 seq_len 和 num_heads 换回来
        attention_output = attention_output.transpose(1, 2)

        # 然后，把最后两个维度合并
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model).contiguous()

        # 最终投影
        return self.wo(attention_output)
