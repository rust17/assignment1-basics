import torch
import torch.nn as nn
from ..layers.embedding import Embedding
from .transformer_block import TransformerBlock
from ..layers.normalization import RMSNorm
from ..layers.linear import Linear

class TransformerLm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
            ) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model=d_model)

        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 根据 Figure 1 这张总装图。整个模型的流程是：
        # 1. 输入 Token ID (batch_size, seq_len)
        # 2. Token Embedding: 把它变成 (batch_size, seq_len, d_model) 的向量。
        # 3. 循环 num_layers 次: 把向量序列依次穿过我们刚刚完成的 TransformerBlock。
        # 4. 最终的 Norm: 对 num_layers 个 Block 的输出做最后一次 RMSNorm。
        # 5. 输出投影 (LM Head): 用一个 Linear 层，把 d_model 维向量投影到 vocab_size 维，得到 logits。
        # 6. Softmax: 把 logits 转换成概率分布。(在训练时，计算交叉熵损失的函数通常会内置 softmax，所以你自己的模块可以只输出 logits；但在推理时，你可能需要手动加 softmax)

        # TransformerBlock 需要 positions 参数。这个参数应该在 TransformerLM 的 forward 方法里创建，因为它取决于输入 x 的序列长度
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)

        x = self.token_embeddings(x)

        # nn.ModuleList 就像一个普通的 Python 列表，需要遍历它里面的每一个模块
        for block in self.layers:
            x = block(x, positions)

        logits = self.lm_head(self.ln_final(x))

        return logits
