import os
import requests
import tiktoken
import numpy as np

# 打开数据文件tang_poet.txt
input_file_path = 'tang_poet.txt'
# 核心添加：指定UTF-8编码读取文件，兼容中文/特殊字符，添加错误处理
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()

# 按照9:1的比例将数据划分为训练集和测试集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 使用GPT-2 BPE的编码方式对数据进行编码分词
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 将训练集和测试集另存为train.bin和val.bin二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__),'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__),'val.bin'))