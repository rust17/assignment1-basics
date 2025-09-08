"""
在 TinyStories 数据集上训练字节级 BPE 分词器

TinyStories 数据集训练资源消耗：
耗时：1.1 min
内存使用：103.1 MB
最长标记：b' accomplishment'
最长标记长度：15

OpenWebText 数据集训练资源消耗：
OpenWebText 文本太大，卡在中间了，暂时先不执行 OpenWebText
$ uv run cs336_basics/demo3/train_bpe.py
Processed and merged results from chunk 1/28...
Processed and merged results from chunk 2/28...
Processed and merged results from chunk 3/28...
Processed and merged results from chunk 4/28...
"""

import time
import psutil
import pickle
from cs336_basics.BPE.demo3.bpe_processing import train_bpe_tokenizer

def analyze_performance(input_path: str, vocab_output_path: str, merges_output_pth: str):
    start = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss

    vocabs, merges = train_bpe_tokenizer(
        input_path=input_path,
        vocab_size=32000,
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

    with open(vocab_output_path, "wb") as f:
        pickle.dump(vocabs, f)

    with open(merges_output_pth, "wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    analyze_performance('/home/circle/code/assignment1-basics/data/owt_train.txt', 'owt_vocab.pkl', 'owt_merges.pkl')