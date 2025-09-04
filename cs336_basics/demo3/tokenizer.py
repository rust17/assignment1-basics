import regex as re
from typing import Iterable, Iterator
import pickle

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=list[str]|None):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        self.merges_map = {pair: i for i, pair in enumerate(merges)}
        # 创建反向词汇表：bytes -> id
        self.vocab_rev = {v: k for k, v in vocab.items()}
        # 创建 special_tokens 到 id 的映射
        self.special_tokens_map = {}
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes in self.vocab_rev:
                self.special_tokens_map[token] = self.vocab_rev[token_bytes]

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从文件加载词汇表和合并规则来创建 Tokenizer 实例

        Args:
            vocab_filepath: 词汇表文件路径 (pickle格式)
            merges_filepath: 合并规则文件路径 (pickle格式)
            special_tokens: 特殊标记列表
        """
        # 加载词汇表
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        # 加载合并规则
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        # 预分词 - 保留 special_tokens，处理重叠情况
        words = []
        i = 0
        while i < len(text):
            # 找到最长匹配的 special token
            best_match = None
            best_length = 0

            for token in self.special_tokens:
                if text[i:i+len(token)] == token and len(token) > best_length:
                    best_match = token
                    best_length = len(token)

            if best_match:
                # 找到了 special token
                words.append(best_match)
                i += best_length
            else:
                # 没找到 special token，收集普通文本直到下一个 special token
                start = i
                i += 1
                # 继续收集字符，直到遇到 special token 或文本结束
                while i < len(text):
                    found_special = False
                    for token in self.special_tokens:
                        if text[i:i+len(token)] == token:
                            found_special = True
                            break
                    if found_special:
                        break
                    i += 1

                # 添加普通文本部分
                if start < i:
                    words.append(text[start:i])

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        all_ids = []

        for word in words:
            # 如果是 special token，直接转换为对应的 ID
            if word in self.special_tokens_map:
                all_ids.append(self.special_tokens_map[word])
                continue

            # 转成字节列表：[b't', b'h', b'e']
            pre_tokens = []
            for match in re.finditer(PAT, word):
                pre_tokens.append(list(bytes([i]) for i in match.group().encode('utf-8')))

            # 对每个 token 应用合并规则
            tokens = []
            for token in pre_tokens:
                # 循环应用合并规则
                while len(token) > 1:
                    # 1. 找到所有相邻对
                    pairs = []
                    for i in range(len(token) - 1):
                        pairs.append((token[i], token[i+1]))

                    # 2. 查找这些对在 merges_map 里的优先级，找到优先级最高的
                    best_pair = None
                    best_idx = None
                    highest_priority = float('inf')  # 优先级越小越高

                    for i, pair in enumerate(pairs):
                        priority = self.merges_map.get(pair)
                        if priority is not None and priority < highest_priority:
                            highest_priority = priority
                            best_pair = pair
                            best_idx = i

                    # 3. 如果找不到优先级，结束循环
                    if best_pair is None:
                        break

                    # 4. 如果找到了优先级，则执行合并
                    # 合并 best_pair 在位置 best_idx
                    new_token = []
                    i = 0
                    while i < len(token):
                        if i == best_idx and i + 1 < len(token) and (token[i], token[i+1]) == best_pair:
                            # 执行合并
                            merged = token[i] + token[i+1]
                            new_token.append(merged)
                            i += 2  # 跳过下一个元素
                        else:
                            new_token.append(token[i])
                            i += 1

                    token = new_token

                tokens.extend(token)

            # 5. 拿到合并后的序列，在词汇表里查找，转成ID列表
            for token in tokens:
                if token in self.vocab_rev:
                    all_ids.append(self.vocab_rev[token])
                else:
                    # 如果找不到 token，这里可能需要处理未知 token
                    raise ValueError(f"Unknown token: {token}")

        return all_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 遍历输入的迭代器
        for chunk in iterable:
            # 对于每一块文本，调用 encode 方法
            ids = self.encode(chunk)

            # 把结果 yield 返回
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        res = []
        for id in ids:
            if id in self.vocab:
                res.append(self.vocab[id])

        # 将所有字节拼接成一个字符串，然后解码成 utf-8 返回
        all_bytes = b''.join(res)
        return all_bytes.decode('utf-8', errors='ignore')
