"""
在 TinyStories 数据集上训练字节级 BPE 分词器
使用最大词汇量 10,000，添加 TinyStories 特殊标记
"""

import time
import psutil
from cs336_basics.demo3.bpe_processing import train_bpe_tokenizer

def analyze_performance():
    input_path = 'your/path/tinystories_train.txt'

    start = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss

    vocabs, merges = train_bpe_tokenizer(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=['<|endoftext|>']
    )

    during = time.time() - start
    print(f"耗时：{during/60:.1f} min")

    end_memory = process.memory_info().rss
    cost_memory = (end_memory-start_memory) / 1024 / 1024
    print(f"内存使用：{cost_memory:.1f} MB")

    longest_token = b''
    for _, token_bytes in vocabs.items():
        if len(token_bytes) > len(longest_token):
            longest_token = token_bytes

    print(f"最长标记：{longest_token}")
    print(f"最长标记长度：{len(longest_token)}")

if __name__ == "__main__":
    analyze_performance()