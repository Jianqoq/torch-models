import ijson
from ijson.common import ObjectBuilder
import re
import time
import torch
from langdetect import detect
from torch.nn import Module, MultiheadAttention, Linear, Embedding, LayerNorm, ReLU, \
    ModuleList
from language_tool_python import LanguageTool
from layers import printProcess
import nltk
import multiprocessing as mp
import json
import mwparserfromhell
import xml.etree.ElementTree as ET
from layers import Preprocess
from blingfire import *

torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Bert(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, max_length, num_layers, n_segments=2):
        super().__init__()
        self.tokenEmbedding = Embedding(corpus, embedding_dim)
        self.segmentEmbedding = Embedding(n_segments, embedding_dim)
        self.position_embedding = Embedding(max_length, embedding_dim)
        self.encoder = ModuleList([Encoder(embedding_dim, hidden_size, num_head) for _ in range(num_layers)])
        self.num_heads = num_head
        self.seq_len = 512

    def forward(self, questions, word_id, mask0, prob):
        segments = torch.ones(questions.shape, dtype=torch.int, device=device) * mask0
        prob_mask = torch.bernoulli(torch.full(mask0.shape, prob, device=device) * mask0).long()
        not_prob_mask = torch.logical_not(prob_mask)
        mask_word_mask = prob_mask * word_id["[MASK]"]
        questions = not_prob_mask * questions + mask_word_mask
        positions = torch.arange(questions[0].shape[0], device=device).unsqueeze(0).repeat(questions.shape[0],
                                                                                           1) * mask0
        mask = (mask0 != False).unsqueeze(1).repeat(self.num_heads, self.seq_len, 1)

        word_vec = self.tokenEmbedding.forward(questions.long())
        segment = self.segmentEmbedding.forward(segments)
        position = self.position_embedding.forward(positions)
        inp = word_vec + segment + position
        for i in self.encoder:
            inp = i.forward(inp, mask)
        return inp


class Encoder(Module):
    def __init__(self, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm2 = LayerNorm(embedding_dim)

    def forward(self, word_vec, mask):
        x, weights = self.multiHeadAttention(word_vec, word_vec, word_vec, attn_mask=mask)
        o = self.layerNorm.forward(x + word_vec)
        ff_result = self.feedForward.forward(o)
        o2 = self.layerNorm2.forward(ff_result + o)
        return o2


class FeedForward(Module):
    def __init__(self, in_feature, out_feature, hidden_feature):
        super().__init__()
        self.linear1 = Linear(in_feature, hidden_feature)
        self.linear2 = Linear(hidden_feature, out_feature)
        self.relu = ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


if __name__ == "__main__":
    print("Offset 209, length 4, Rule ID: COMMA_COMPOUND_SENTENCE\nMessage: Use a comma before \u2018and\u2019 if it connects two independent clauses (unless they are closely connected and short).\nSuggestion: , and\n...d by the alienation caused by capitalism and it prevents humans from living a joyful...\n")
    pass
    # _pad_index = 0
    # _vocab_size = len(_word_id)
    # _embedding_dim = 128
    # _hidden_size = 516
    # _num_head = 8
    # _out_dim = 512
    # max_epoch = 50
    # batch = 10
    # _num_layers = 2
    # transformer = Bert(_vocab_size, _embedding_dim, _hidden_size, _num_head, 512, _num_layers)
    # transformer.to(device)
    # with h5py.File('data.h5', 'r') as f:
    #     # 将 NumPy 数组转换回 PyTorch tensor
    #     a_tensor = torch.from_numpy(f["article_1"][:]).to(device)
    # mask1 = a_tensor != 0  # random mask for guessing
    #
    #
    # result = transformer.forward(a_tensor, _word_id, mask1.to(device), 0.2)
    # transformer.train()
    # optimizer = torch.optim.Adam(transformer.parameters())
    # trainer = Train(transformer, optimizer)
    # trainer.PYTORCH_train(train_questions, train_answer, test_questions, test_answer, batch,
    #                       max_epoch, word_id, id_word, log_dir="runs", log=True,
    #                       log_file_name="Transformer")
