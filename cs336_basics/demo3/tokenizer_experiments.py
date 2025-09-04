"""
对分词器的实验
"""

from cs336_basics.demo3.tokenizer import Tokenizer
from cs336_basics.demo3.train_bpe import analyze_performance
import time, os
from pathlib import Path
import numpy as np

def split_document(input_path: str, special_token: str, max_results=10):
    """
    拆分文件返回前 10 个部分
    """
    results = []

    with open(input_path, 'r') as file:
        buffer = ''

        for line in file:
            buffer += line

            if special_token in buffer:
                parts = buffer.split(special_token)

                for part in parts[:-1]:
                    if part.strip():
                        results.append(part.strip())
                        if len(results) >= max_results:
                            return results

                buffer = parts[-1]

        if buffer.strip() and len(results) < max_results:
            results.append(buffer.strip())

    return results[:max_results]

def extract_chunk(input_path: str, output_file: str, chunk_size_mb=100):
    chunk_size = chunk_size_mb * 1024 * 1024

    with open(input_path, 'rb') as file_in:
        with open(output_file, 'wb') as file_out:
            data = file_in.read(chunk_size)
            file_out.write(data)

    print(f"成功截取 {chunk_size_mb}MB 数据到 {output_file}")

def calculate_compression():
    """
    从 TinyStories 和 OpenWebText 中各选取 10 个文档。使用训练的 TinyStories 和 OpenWebText 分词器（分别具有 10K 和 32K 的词汇量），将这些采样文档编码为整数 ID。每个分词器的压缩比（字节/标记）是多少？

    4.265895953757226
    4.080246913580247
    4.071428571428571
    4.481675392670157
    4.202643171806168
    4.059880239520958
    3.9
    4.027522935779817
    4.056390977443609
    4.2439024390243905
    """

    original_filepath = '/home/circle/code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_filepath = '/home/circle/code/assignment1-basics/tinystories_vocab.pkl'
    merges_filepath = '/home/circle/code/assignment1-basics/tinystories_merges.pkl'

    texts = split_document(original_filepath, '<|endoftext|>')
    for text in texts:
        tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, ['<|endoftext|>'])
        text_tokens = len(tokenizer.encode(text))
        text_bytes = len(text.encode('utf-8'))

        print(text_bytes / text_tokens)

def calculate_throughput():
    """
    估计你分词器的吞吐量（例如，以字节/秒计）

    吞吐量：1.52 MB/s
    """

    from_filepath = '/home/circle/code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    to_filepath = "/home/circle/code/assignment1-basics/data/TinyStoriesV2-GPT4-train_100M.txt"
    vocab_filepath = '/home/circle/code/assignment1-basics/tinystories_100M_vocab.pkl'
    merges_filepath = '/home/circle/code/assignment1-basics/tinystories_100M_merges.pkl'

    if not Path(to_filepath):
        extract_chunk(from_filepath, to_filepath)

    if not Path(vocab_filepath) or not Path(merges_filepath):
        analyze_performance(to_filepath, vocab_filepath, merges_filepath)

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)
    count = 0
    start = time.time()
    with open(to_filepath, 'r') as file:
        for _ in tokenizer.encode_iterable(file):
            count += 1
    during = time.time() - start

    with open(to_filepath, 'r') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()

    print(f"吞吐量：{file_size / 1024 / 1024 / during:.2f} MB/s")

def encode_to_np():
    """
    使用 TinyStories 分词器，将相应的训练和开发数据集编码为一系列整数标记 ID。我们稍后将使用这些 ID 来训练我们的语言模型。我们建议将标记 ID 序列化为一个 uint16 数据类型的 NumPy 数组。为什么 uint16 是一个合适的选择？

    执行时间：25 min
    uint16 的范围是 0～65535，而词汇表大小是：10000(TinyStories)、32000(OpenWebText)，所以 uint16 是合适的
    """
    filepath = "/home/circle/code/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_filepath = '/home/circle/code/assignment1-basics/tinystories_vocab.pkl'
    merges_filepath = '/home/circle/code/assignment1-basics/tinystories_merges.pkl'

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, ['<|endoftext|>'])
    ids = []
    with open(filepath, 'r') as file:
        for id in tokenizer.encode_iterable(file):
            ids.append(id)

    np_ids = np.array(ids, dtype=np.uint16)
    print(f"np_ids：{np_ids}")

if __name__ == '__main__':
    # calculate_compression()
    # calculate_throughput()
    encode_to_np()
