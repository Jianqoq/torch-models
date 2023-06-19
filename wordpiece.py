import json

from tokenizers.implementations import BertWordPieceTokenizer

from layers import printProcess
import h5py
import numpy as np

# 将句子存储到HDF5文件中
char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)]
char_list = char_list[:char_list.index('FU')+1]
total = len(char_list) * 100
count = 0
# p = printProcess()
# p.add_bar('progress')
# for chars in char_list:
#     for i in range(100):
#         with open(fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json_sentence.json", "r",
#                   encoding='utf-8') as fp:
#             dictionary = json.load(fp)
#             sentence = list(dictionary.values())
#             count += 1
#             p.print_result((count, total))
#             ls = []
#             for k in sentence:
#                 for sent in k:
#                     ls.append(sent + '\n')
#             with open(fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.sentence.txt", "w",
#                       encoding='utf-8') as f:
#                 f.writelines(ls)
from tokenizers import Tokenizer, models, trainers
with open("saved_word_id.json", "r", encoding='utf-8') as fp:
    word_id = json.load(fp)
files = [fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.sentence.txt" for i in range(100) for chars in char_list]

tokenizer = BertWordPieceTokenizer()

# 训练分词器
tokenizer.train(files, vocab_size=30522)

# 保存分词器和词汇表
tokenizer.save_model("custom")

