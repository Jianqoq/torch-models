import json
import sys

import torch
from torch.nn import Module, MultiheadAttention, Linear, Embedding, LayerNorm, ReLU, \
    ModuleList

from layers import Train

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
        self.encoder = Encoder(embedding_dim, hidden_size, num_head)
        self.linear = Linear(_embedding_dim, corpus)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.num_heads = num_head

    def forward(self, questions, word_id, mask0, prob):
        questions = questions.long()
        segments = torch.ones(questions.shape, dtype=torch.long, device=device) * mask0  # mask0 non pad mask
        prob_mask = torch.bernoulli(torch.full(mask0.shape, prob, device=device) * mask0).bool()
        not_prob_mask = torch.logical_not(prob_mask)
        mask_word_mask = prob_mask * word_id["[MASK]"]
        questions_processed = not_prob_mask * questions + mask_word_mask
        positions = torch.arange(questions_processed[0].shape[0], device=device).unsqueeze(0).repeat(questions_processed.shape[0], 1) * mask0
        mask = (mask0 == True).unsqueeze(1).repeat(self.num_heads, questions_processed.shape[1], 1)

        word_vec = self.tokenEmbedding.forward(questions_processed.long())
        segment = self.segmentEmbedding.forward(segments)
        position = self.position_embedding.forward(positions)
        inp = word_vec + segment + position
        inp = self.encoder.forward(inp, mask)
        linear = self.linear.forward(inp)
        after = torch.permute(linear, (0, 2, 1))
        # result = prob_mask * questions
        loss = self.loss_fn.forward(after, questions)
        return loss


class Encoder(Module):
    def __init__(self, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm2 = LayerNorm(embedding_dim)

    def forward(self, word_vec, mask):
        x, weights = self.multiHeadAttention.forward(word_vec, word_vec, word_vec, attn_mask=mask)
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
    with open('saved_word_id.json', 'r', encoding='utf-8') as f:
        _word_id = json.load(f)
    with open('saved_id_word.json', 'r', encoding='utf-8') as f:
        _id_word = json.load(f)
    with open(r'C:\Users\123\PycharmProjects\words2\AA\wiki_00.json', 'r', encoding='utf-8') as f:
        _words = json.load(f)['words']
    _pad_index = 0
    _vocab_size = len(_word_id)
    _embedding_dim = 128
    _hidden_size = 516
    _num_head = 8
    _out_dim = 512
    max_epoch = 50
    batch = 10
    _num_layers = 1
    for article in range(len(_words)):
        for sentence in range(len(_words[article])):
            for idx3, word in enumerate(_words[article][sentence]):
                _words[article][sentence][idx3] = _word_id[word]
            _words[article][sentence] = torch.tensor(_words[article][sentence], device=device, dtype=torch.int32)
        _words[article].insert(0, torch.tensor([0] * (len(max(_words[article], key=len)) + 1), device=device, dtype=torch.int32))
        _words[article] = torch.nn.utils.rnn.pad_sequence(_words[article], batch_first=True)
    bert = Bert(_vocab_size, _embedding_dim, _hidden_size, _num_head, 512, _num_layers)
    bert.to(device=device, dtype=torch.float32)
    mask1 = _words[1] != 0  # random mask for guessing
    bert.train()
    optimizer = torch.optim.Adam(bert.parameters())
    trainer = Train(bert, optimizer)
    trainer.add_bar('Epoch', 'Iter')
    trainer.add_metrics('loss')
    trainer.custom_train(_words, _words, batch,
                         max_epoch, _word_id, _id_word, log_dir="bert", log=True,
                         log_file_name="Bert")