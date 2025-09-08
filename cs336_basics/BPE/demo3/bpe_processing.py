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
import os
import multiprocessing
from collections import defaultdict

def find_split_points(input_path: str, nums_of_split: int, special_token: str) -> list[int]:
    """
    找到大文件的分割点，确保在每个分割点处都是完整的特殊标记

    参数:
        input_path: 输入文件路径
        nums_of_split: 分割的块数（进程数量）
        special_token: 用于分割的特殊标记

    返回:
        包含每个分割点偏移量的列表
    """

    file_size = os.path.getsize(input_path)

    # 计算理论分割点
    splits = [i * file_size // nums_of_split for i in range(nums_of_split + 1)]

    actual_splits = [0]

    special_token_bytes = special_token.encode('utf-8')

    with open(input_path, 'rb') as f:
        for i in range(1, nums_of_split):
            # 定位到理论分割点
            f.seek(splits[i])

            # 读取足够的数据来查找特殊标记（这里读取 64 KB）
            data = f.read(65535)

            # 查找任意特殊标记的位置
            pos = data.find(special_token_bytes)

            if pos != -1:
                # 找到特殊标记，计算实际分割点
                actual_splits.append(splits[i] + pos)
            else:
                # 如果没找到特殊标记，寻找最近的换行符作为分割点
                newline_pos = data.find(b'\n')
                if newline_pos != -1:
                    actual_splits.append(splits[i] + newline_pos + 1)  # 换行符后
                else:
                    actual_splits.append(splits[i])

        actual_splits.append(file_size)

    return actual_splits

def regex_pretokenize(args: tuple[str, int, int, list[str]]) -> dict[tuple, int]:
    """
    预处理语料库：使用正则表达式进行预分词，转换为字节序列

    参数：
    - text: 输入的原始文本
    - special_tokens: 特殊token列表，如 ["<|endoftext|>", "<|startoftext|>"]

    返回值：
    - pre_token_freqs: 字节序列 token 到频率的映射，如 {(b'h', b'e', b'l', b'l', b'o'): 3}

    思路：
    1. 首先按特殊 token 分割文本，保留特殊 token
    2. 对每个文本片段使用正则表达式进行预分词
    3. 将每个预 token 转换为 UTF-8 字节序列的元组
    4. 统计每个字节序列的出现频率
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:  # 以二进制模式打开避免 UTF-8 解码问题
        f.seek(start)
        data = f.read(end - start)

    # 确保在 UTF-8 字符边界处截断，使用错误处理
    text = data.decode('utf-8', errors='ignore')

    delimiter = "|".join(map(re.escape, special_tokens))
    words = re.split(delimiter, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_token_freqs = defaultdict(int)
    for word in words:
        if word.strip():
            for match in re.finditer(PAT, word):
                token = tuple(bytes([i]) for i in match.group().encode('utf-8'))
                pre_token_freqs[token] += 1

    return pre_token_freqs

def get_stats(token_freqs: dict[tuple, int]) -> tuple[dict[tuple, int], dict[tuple, set]]:
    """
    统计所有相邻字节对的出现频率

    参数：
    - token_freqs: token 序列到频率的映射，格式如 {(b'h', b'e', b'l', b'l', b'o'): 5}

    返回值：
    - pair_freqs: 字节对到总频率的映射，如 {(b'h', b'e'): 10, (b'e', b'l'): 15}
    - pair_tokens: 字节对到包含该字节对的所有 token 集合的映射

    思路：
    1. 遍历所有 token 序列
    2. 对每个序列，统计相邻字节对的出现次数
    3. 同时记录每个字节对出现在哪些 token 中，便于后续合并时快速定位
    """
    pair_freqs = defaultdict(int)
    pair_tokens = defaultdict(set)

    for token, freq in token_freqs.items():
        token_set = tuple(token)
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_freqs[pair] += freq
            pair_tokens[pair].add(token_set)

    return pair_freqs, pair_tokens

def merge(token: list, top_pair: tuple, pair_freqs: dict[tuple, int], pair_tokens: dict[tuple, set], pre_token_freqs: dict[tuple, int]):
    """
    合并指定 token 中的最频繁字节对

    参数：
    - token: 要合并的 token 序列
    - top_pair: 要合并的字节对 (byte1, byte2)
    - pair_freqs: 字节对频率统计字典
    - pair_tokens: 字节对到包含它的 token 集合的映射
    - pre_token_freqs: token 频率统计字典

    功能：
    1. 在指定 token 中合并 top_pair 字节对
    2. 更新 token 频率统计
    3. 更新字节对频率统计（移除旧的，添加新的）
    """
    freq = pre_token_freqs.pop(token)
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

    new_token_merged = tuple(new_token_merged)
    pre_token_freqs[new_token_merged] = freq
    # 为当前 token 移除旧的统计信息
    for j in range(len(token) - 1):
        pair = (token[j], token[j+1])
        pair_freqs[pair] -= freq
        pair_tokens[pair].discard(token)

    # 为新 token 添加统计信息
    for j in range(len(new_token_merged) - 1):
        new_pair = (new_token_merged[j], new_token_merged[j+1])
        pair_freqs[new_pair] += freq
        pair_tokens[new_pair].add(new_token_merged)

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器的主函数

    参数：
    - input_path: 训练语料文件路径
    - vocab_size: 目标词汇表大小（包含基础256字节+特殊 token）
    - special_tokens: 特殊token列表，如 ["<|endoftext|>"]

    返回值：
    - vocabs: 词汇表字典，{token_id: bytes}，如 {0: b'\\x00', 256: b'the', ...}
    - merges: 合并规则列表，[(byte1, byte2), ...]，按合并顺序排列

    算法流程：
    1. 初始化词汇表：
       - 前 256 个位置：所有可能的字节值 0x00-0xFF
       - 后续位置：特殊 token（如果有）
    2. 预处理语料库，获取字节序列 token 及其频率统计
    3. 迭代合并过程（vocab_size - 初始 vocab 大小 次）：
       a. 统计所有相邻字节对的频率
       b. 选择最频繁的字节对（频率相同时选择字典序较大的）
       c. 合并该字节对，创建新 token 并分配 ID
       d. 更新词汇表和合并规则列表
       e. 增量更新统计信息（仅处理受影响的 token）
    4. 返回最终的词汇表和按顺序的合并规则列表

    性能优化：
    - 使用增量更新避免重新计算所有统计信息
    - 维护字节对到包含 token 的映射，快速定位需要更新的 token
    """
    vocabs = {i: bytes([i]) for i in range(256)}
    for id, token in enumerate(special_tokens, start=256):
        vocabs[id] = token.encode('utf-8')

    split_points = find_split_points(input_path, multiprocessing.cpu_count(), "<|endoftext|>")

    args = [(input_path, split_points[i], split_points[i+1], special_tokens) for i in range(len(split_points) - 1)]

    pre_token_freqs = defaultdict(int)
    with multiprocessing.Pool() as pool:
        pre_iterator = pool.imap_unordered(regex_pretokenize, args)

        for i, freqs in enumerate(pre_iterator):
            for key, value in freqs.items():
                pre_token_freqs[key] += value
            print(f"  Processed and merged results from chunk {i+1}/{len(args)}...")

    merges = []
    # 初始化统计信息，只计算一次
    pair_freqs, pair_tokens = get_stats(pre_token_freqs)
    print(f"Total unique pairs: {len(pair_freqs)}")

    num_merges = vocab_size - len(vocabs)
    for _ in range(num_merges):
        if i % 100 == 0: # 每 100 次合并打印一次进度
            print(f"  Merge iteration {i}/{num_merges}...")
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
        affected_tokens = list(pair_tokens[top_pair])
        pair_tokens[top_pair].clear()
        for token in affected_tokens:
            merge(token, top_pair, pair_freqs, pair_tokens, pre_token_freqs)

    return (vocabs, merges)

"""
算法复杂度分析：

时间复杂度：
- 预处理阶段: O(N)，其中 N 是输入文本长度
  * 正则表达式分词: O(N)
  * 字节转换和频率统计: O(N)

- 合并迭代阶段: O(V * (A + U))，其中：
  * V = vocab_size - 256 - len(special_tokens)（需要学习的 merge 数量）
  * A = 平均每轮受影响的 token 数量
  * U = 每个受影响 token 的平均更新成本

- 每轮合并细节:
  * 找到最频繁字节对: O(P)，P 是当前不同字节对的数量
  * 更新受影响 token: O(A * L)，L 是平均 token 长度
  * 增量更新统计: O(A * L)

- 总体时间复杂度: O(N + V * (P + A * L))，把 (P + A * L) 看成一个与当前词汇状态和文本相关的因子，可以看成是 O(N + V * M)，坏的情况接近于 O(N²)，平均优于 O(N²)

空间复杂度：
- 词汇表存储: O(V)，最终 vocab_size 个条目
- 原始 token 序列: O(T)，所有字节序列的存储
- 字节对频率统计: O(P)，不同字节对的数量（通常 P << T）
- 字节对到 token 映射: O(P * A)，每个字节对对应的 token 集合
- 合并规则列表: O(V)，存储所有 merge 操作

- 总体空间复杂度: O(T + P * A + V)

性能特征：
1. **最坏情况**: 语料库包含大量独特的长 token，每轮只能合并很少的实例
2. **最优情况**: 语料库高度重复，每轮可以合并大量实例
3. **实际表现**: 通常随着迭代进行，受影响 token 数量(A)会逐渐减少
"""