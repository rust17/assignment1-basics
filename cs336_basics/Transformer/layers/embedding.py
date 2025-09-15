import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        # 我们的词汇表大小是 num_embeddings，每个词要被映射成长度为 embedding_dim 的向量
        # 查询表就是一个矩阵，每一行对应词汇表里的一个词（一个 token ID），每一行都是一个词的向量表示，向量维度是 embedding_dim
        # 这个查询表矩阵就是模型需要学习的东西，我们需要用 Pytorch 的 nn.Parameter 来包装
        tensor_embedding = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.embedding_matrix = nn.Parameter(tensor_embedding)

        # 实现公式：Embedding: N(µ = 0, σ2 = 1) truncated at [−3, 3]
        # 初始化成均值是 0，方差为 1 的截断正态分布
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # forward 方法接收的输入 token_ids 是一个整数张量，形状是 (batch_size, sequence_length)
        # 我们需要用这些整数 ID，去我们创建好的 self.embedding_matrix 里“查表取值”
        # 在 PyTorch 里，如果 E 是我们的 embedding 矩阵，ids 是我们的 ID 张量，直接用 E[ids] 就可以完成取值操作
        # PyTorch 会自动处理 token_ids 的批处理和序列维度，对里面的每一个 ID 都去 embedding_matrix 里找到对应的那一行向量，然后把结果按照 token_ids 原来的形状 (batch_size, sequence_length) 组织起来，只不过最后一维从一个整数 ID 变成了一个 embedding_dim 长度的向量。
        # 所以输出的形状会是 (batch_size, sequence_length, embedding_dim)，这正是下一层 Transformer Block 想要的输入形状
        return self.embedding_matrix[token_ids]
