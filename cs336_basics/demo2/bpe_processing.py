"""
BPE（Byte Pair Encoding）分词器训练的完整代码实现

主要思路：
1. 预处理：将输入文本分割成预 token，处理特殊 token
2. 初始化：创建基础词汇表（256 个字节 + 特殊 token）
3. 统计：统计相邻字节对的出现频率
4. 合并：重复合并最频繁的字节对，直到达到目标词汇表大小
5. 输出：返回词汇表和合并规则
"""

import regex as re
from collections import defaultdict

def regex_pretokenize(text: str, special_tokens: list[str]) -> list[list[int]]:
    """
    预处理语料库：读取文件，进行预分词，转换为字节序列

    思路：
    1. 读取输入文件的文本内容
    2. 使用正则表达式进行预分词，处理特殊 token
    3. 将每个预 token 转换为字节序列（UTF-8编码）
    4. 返回字节序列列表
    """
    delimiter = "|".join(map(re.escape, special_tokens))
    words = re.split(f'({delimiter})', text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens = []
    for word in words:
        if word in special_tokens:
            pre_tokens.append(list(word.encode('utf-8')))
        elif word.strip():
            for match in re.finditer(PAT, word):
                pre_tokens.append([bytes([i]) for i in match.group().encode('utf-8')])

    return pre_tokens

def get_stats(tokens: list[list[bytes]]) -> dict[tuple, int]:
    """
    统计所有相邻字节对的出现频率

    思路：
    1. 遍历所有 token 序列
    2. 对每个序列，统计相邻字节对的出现次数
    3. 返回字节对到频率的映射
    """
    pair_freqs = defaultdict(int)

    for token in tokens:
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_freqs[pair] += 1

    return pair_freqs

def merge(token: list, top_pair: tuple, pair_freqs: dict[tuple, int]):
    # 合并当前 token
    new_token_merged = []
    j = 0
    while j < len(token):
        if (j < len(token) - 1 and
            token[j] == top_pair[0] and
            token[j+1] == top_pair[1]):
            new_token_merged.append(top_pair[0] + top_pair[1])
            j += 2
        else:
            new_token_merged.append(token[j])
            j += 1

    # 为当前 token 移除旧的统计信息
    for j in range(len(token) - 1):
        pair = (token[j], token[j+1])
        pair_freqs[pair] -= 1

    # 为新 token 添加统计信息
    for j in range(len(new_token_merged) - 1):
        new_pair = (new_token_merged[j], new_token_merged[j+1])
        pair_freqs[new_pair] += 1

    return new_token_merged

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器的主函数

    算法流程：
    1. 初始化词汇表（256字节 + 特殊 token）
    2. 预处理语料库，获取 token 序列
    3. 迭代合并过程：
       a. 统计相邻字节对频率，考虑字典序大小
       b. 找到最频繁的字节对
       c. 合并该字节对，创建新 token
       d. 更新词汇表和合并规则
    4. 重复步骤 3，直到达到目标词汇表大小
    5. 返回最终的词汇表和合并规则列表
    """
    vocabs = {i: bytes([i]) for i in range(256)}
    for id, token in enumerate(special_tokens, start=256):
        vocabs[id] = token.encode('utf-8')

    with open(input_path, "r", encoding="utf-8") as f: text = f.read()

    pre_tokens = regex_pretokenize(text, special_tokens)

    merges = []
    # 初始化统计信息，只计算一次
    pair_freqs = get_stats(pre_tokens)

    num_merges = vocab_size - len(vocabs)
    for _ in range(num_merges):
        # 优化：一次遍历找到最频繁的字节对，考虑字典序
        top_pair = None
        max_freq = 0
        for pair, freq in pair_freqs.items():
            if freq > max_freq or (freq == max_freq and (top_pair is None or pair > top_pair)):
                max_freq = freq
                top_pair = pair

        merges.append((top_pair[0], top_pair[1]))

        vocabs[len(vocabs)] = top_pair[0] + top_pair[1]

        # 更新 pre_tokens 并增量更新统计信息
        for index, token in enumerate(pre_tokens):
            if token.count(top_pair[0]) > 0 or token.count(top_pair[1]) > 0:
                pre_tokens[index] = merge(token, top_pair, pair_freqs)

    return (vocabs, merges)

"""
算法复杂度分析：

时间复杂度：
- 预处理: O(N)，其中 N 是输入文本长度
- 每轮合并: O(M)，其中 M 是当前 token 序列总长度
- 总体: O(V * M)，其中V是要学习的 merge 数量

空间复杂度：
- 词汇表: O(V)
- Token序列: O(M)
- 统计信息: O(P)，其中 P 是不同字节对的数量

优化建议：
1. 对于大型语料库，可以考虑分块处理
2. 可以使用更高效的数据结构来加速字节对统计
3. 可以并行化不同文件的预处理过程
4. 考虑使用内存映射文件来处理超大文件
"""