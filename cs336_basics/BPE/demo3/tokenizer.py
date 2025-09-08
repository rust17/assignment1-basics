import regex as re
from typing import Iterable, Iterator
import pickle

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=list[str]|None):
        self.vocab = vocab
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.merges_map = {pair: i for i, pair in enumerate(merges)}
        # 创建反向词汇表：bytes -> id
        self.vocab_rev = {v: k for k, v in vocab.items()}
        # 创建 special_tokens 到 id 的映射
        self.special_tokens_map = {}
        for token in self.special_tokens:
            self.special_tokens_map[token] = self.vocab_rev[token.encode('utf-8')]

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
        pattern = '|'.join(map(re.escape, self.special_tokens))
        words = re.split(f"({pattern})", text) if self.special_tokens else [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        all_ids = []

        for word in words:
            # 如果是 special token，直接转换为对应的 ID
            if word in self.special_tokens_map:
                all_ids.append(self.special_tokens_map[word])
                continue

            # 转成字节列表：[b't', b'h', b'e']
            pre_tokens = [
                [bytes([i]) for i in match.group().encode('utf-8')]
                for match in re.finditer(PAT, word)
            ]

            # 对每个 token 应用合并规则
            for pre_token in pre_tokens:
                # 循环应用合并规则
                while len(pre_token) > 1:
                    # 1. 找到所有相邻对
                    pairs = [(pre_token[i], pre_token[i+1]) for i in range(len(pre_token) - 1)]

                    # 2. 查找这些对在 merges_map 里的优先级，找到优先级最高的
                    best_pair = None
                    highest_priority = float('inf')  # 优先级越小越高

                    for i, pair in enumerate(pairs):
                        priority = self.merges_map.get(pair)
                        if priority is not None and priority < highest_priority:
                            highest_priority = priority
                            best_pair = pair

                    # 3. 如果找不到优先级，结束循环
                    if best_pair is None:
                        break

                    # 4. 如果找到了优先级，则执行合并
                    i = 0
                    new_token = []
                    while i < len(pre_token):
                        if i + 1 < len(pre_token) and (pre_token[i], pre_token[i+1]) == best_pair:
                            # 执行合并
                            merged = pre_token[i] + pre_token[i+1]
                            new_token.append(merged)
                            i += 2
                        else:
                            new_token.append(pre_token[i])
                            i += 1
                    pre_token = new_token

                for token in pre_token:
                    if token in self.vocab_rev:
                        all_ids.append(self.vocab_rev[token])

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
