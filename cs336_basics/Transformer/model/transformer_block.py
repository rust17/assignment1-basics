import torch
import torch.nn as nn
from ..attention.multi_head import MultipleAttention
from ..layers.feedforward import SwiGLU
from ..layers.normalization import RMSNorm
from ..attention.rope import ROPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int):
        super().__init__()

        rope = ROPE(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.attention = MultipleAttention(d_model=d_model, num_heads=num_heads, rope=rope)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # 第一个子层 (Self-Attention sub-layer):
        # 对 x 做 RMSNorm
        # 把归一化后的结果送入 MultiHeadSelfAttention 模块
        # 把 Attention 的输出，和原始的输入 x 相加（残差连接）

        # 第二个子层 (Feed-Forward sub-layer):
        # 4. 对上一步相加后的结果，再做一次 RMSNorm
        # 5. 把归一化后的结果送入 PositionWiseFeedForward 模块
        # 6. 把 FFN 的输出，和第二个子层的输入相加（另一个残差连接）

        # 数据流是这样的：
        # TransformerBlock.forward -> MultiHeadSelfAttention.forward -> ROPE.forward
        x += self.attention(self.rms_norm1(x), token_positions=positions)

        return x + self.ffn(self.rms_norm2(x))
