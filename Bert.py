import json
import os
import sys

import torch
from sklearn.base import BaseEstimator
from torch import optim
from torch.cuda.amp import autocast
from torch.nn import Module, MultiheadAttention, Linear, Embedding, LayerNorm, ModuleList, GELU, Dropout, ReLU
from tokenizers.implementations import BertWordPieceTokenizer
from layers import Train
from transformers import get_linear_schedule_with_warmup
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
COUNT = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Bert(Module, BaseEstimator):
    def __init__(self, embedding_dim, hidden_size, num_head,
                 max_length, num_layers, tokenizer: BertWordPieceTokenizer,
                 n_segments=2, learning_rate=1e-4,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.corpus = tokenizer.get_vocab_size()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.max_length = max_length
        self.num_layers = num_layers
        self.n_segments = n_segments
        self.learning_rate = learning_rate
        self.device = device
        self.tokenEmbedding = Embedding(self.corpus, embedding_dim, device=device)
        self.segmentEmbedding = Embedding(n_segments, embedding_dim, device=device)
        self.position_embedding = Embedding(max_length, embedding_dim, device=device)
        self.encoder = ModuleList(Encoder(embedding_dim, hidden_size, num_head, device) for _ in range(num_layers))
        self.linear = Linear(embedding_dim, self.corpus, device=device)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.num_heads = num_head
        self.vocab_size = self.corpus
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer
        self.to(device)

    def forward(self, masked_sentence, mask0):
        pad = torch.eq(mask0, torch.tensor(False, device=self.device, dtype=torch.bool))
        segments = torch.ones(masked_sentence.shape, dtype=torch.int, device=self.device) * mask0  # mask0 non pad mask
        if len(masked_sentence.shape) > 1:
            positions = torch.arange(masked_sentence[0].shape[0], device=self.device).unsqueeze(0).repeat(
                masked_sentence.shape[0], 1) * mask0
        else:
            positions = torch.arange(masked_sentence.shape[0], device=self.device) * mask0
        word_vec = self.tokenEmbedding.forward(masked_sentence.long())
        segment = self.segmentEmbedding.forward(segments)
        position = self.position_embedding.forward(positions)
        inp = word_vec + segment + position
        for i in self.encoder:
            inp = i.forward(inp, pad)
        return inp
        # return self.linear.forward(inp)

    def generate_predicted_text(self, questions, mask0, prob, to_save, key, monitor=False):
        ques_str = self.tokenizer.decode(questions.tolist())
        prob_mask, not_prob_mask = self.get_prob_mask(mask0, prob)
        masked_sentence = prob_mask * self.tokenizer.token_to_id("[MASK]") + not_prob_mask * questions
        masked_ques_str = self.tokenizer.decode(masked_sentence.tolist(), skip_special_tokens=False)
        with torch.no_grad():
            score = self.forward(masked_sentence, mask0)
            probability = self.softmax.forward(score)
            words = torch.argmax(probability, dim=-1).mul_(prob_mask)
        predicted_str = self.tokenizer.decode([i if i != 0 else 40 for i in words.tolist()])
        if monitor:
            print()
            print(ques_str)
            print(masked_ques_str)
            print(predicted_str)
        to_save[key] = {
            'origin': ques_str,
            'masked': masked_ques_str,
            'predict': predicted_str
        }

    def predict_mask(self, text: str):
        self.eval()
        with autocast():
            text = self.tokenizer.encode(text).ids
            text = torch.nn.functional.pad(torch.tensor(text, device=self.device, dtype=torch.int16),
                                           (0, self.max_length - len(text)))
            prob_mask = text == torch.tensor(self.tokenizer.token_to_id('[MASK]'), device=self.device,
                                             dtype=torch.int16)
            mask0 = text != 0
            with torch.no_grad():
                score = self.forward(text, mask0)
                score = self.linear(score)
                probability = self.softmax.forward(score)
                words = torch.argmax(probability, dim=-1)
                words.mul_(prob_mask)
            predicted_str = dict(enumerate(self.tokenizer.id_to_token(i) for i in words.tolist() if i != 0))
        return predicted_str

    def fit(self, x, **fit_params):
        _batch = fit_params["batch_size"]
        _max_epoch = fit_params["max_epoch"]
        _tokenizer = fit_params["tokenizer"]
        _layer = fit_params["layer"]
        global COUNT
        COUNT += 1
        print(COUNT, 'embedding', self.embedding_dim, 'hidden_size', self.hidden_size, 'num_head', self.num_head)
        self.train()
        optimizr = optim.Adam(self.parameters(), lr=self.learning_rate)
        trainr = Train(self, optimizr)
        trainr.add_bar('Epoch', 'Iter')
        trainr.add_metrics('loss', float)
        trainer.down_stream(batch, max_epoch, layer, log_dir="Bert_down_stream", log=True,
                            log_file_name="Bert_down_stream", monitor=False)

    @staticmethod
    def get_total_steps(files_list, batch_size, _max_epoch):
        def generate():
            for idx, path in enumerate(files_list):
                with open(path, 'r', encoding='utf-8') as f:
                    ids = json.load(f)['ids']
                    for k in range(len(ids) - 1, -1, -1):
                        if len(ids[k]) > 127:
                            del ids[k]
                    print(idx, '/', len(files_list))
                    yield len(ids) // batch_size if len(ids) >= batch_size else 1
        return sum(i for i in generate()) * _max_epoch

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions

    def score(self, X, y=None):
        self.eval()
        article, mask0, path = X[0]
        total_loss = 0
        _count = 0

        def run():
            prob_mask, not_prob_mask = self.get_prob_mask(mask, 0.2)
            masked_sentence = prob_mask * self.tokenizer.token_to_id("[MASK]") + not_prob_mask.mul_(
                batch_question)
            score = self.forward(masked_sentence, mask)
            score.mul_(prob_mask.unsqueeze(-1).expand(-1, -1, self.vocab_size))
            score = torch.permute(score, (0, 2, 1))
            return self.loss_func.forward(score, batch_question.mul_(prob_mask))

        with torch.no_grad():
            article = article.to(device=DEVICE, dtype=torch.long)
            mask0 = mask0.to(device=DEVICE)
            max_iter = len(article) // 5
            for i in range(max_iter):
                start = i * 5 + 1
                batch_question = article[start:(i + 1) * 5 + 1]
                mask = mask0[start:(i + 1) * 5 + 1]
                loss = run()
                total_loss += loss
                _count += 1
        return -float(total_loss / _count)

    def down_stream(self, question, answer, batch_size):
        correct = torch.tensor(0., device=self.device)
        count = 0
        self.eval()
        def run(train):
            try:
                with torch.no_grad():
                    score = self.forward(train, mask.to(device=self.device))
                    score = layer(score)[:, 0, :]
                    score = self.softmax(score)
                    score = torch.argmax(score, dim=-1)
                return score == answers
            except Exception as e:
                raise e
        max_iter = len(question) // batch_size
        for i in range(max_iter):
            start = i * batch_size + 1
            with autocast():
                batch_question = question[start:(i + 1) * batch_size + 1].to(device=self.device)
                answers = answer[start:(i + 1) * batch_size + 1].to(device=self.device)
                mask = batch_question != 0
                corrects = run(batch_question)
                correct += sum(corrects) / len(corrects)
                count += 1
        self.train()
        return correct / count


class Encoder(Module):
    def __init__(self, embedding_dim, hidden_size, num_head, device):
        super().__init__()
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_head, batch_first=True,
                                                     dropout=0.1, device=device)
        self.layerNorm = LayerNorm(embedding_dim, device=device)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size, device)
        self.layerNorm2 = LayerNorm(embedding_dim, device=device)
        self.drop_out = Dropout(0.1)

    def forward(self, word_vec, pad):
        x, weights = self.multiHeadAttention.forward(word_vec, word_vec, word_vec, key_padding_mask=pad)
        o = self.layerNorm.forward(x.add_(self.drop_out(word_vec)))
        ff_result = self.feedForward.forward(o)
        o2 = self.layerNorm2.forward(self.drop_out(ff_result).add_(o))
        return o2


class FeedForward(Module):
    def __init__(self, in_feature, out_feature, hidden_feature, device):
        super().__init__()
        self.linear1 = Linear(in_feature, hidden_feature, device=device)
        self.linear2 = Linear(hidden_feature, out_feature, device=device)
        self.relu = GELU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class FeedForward2(Module):
    def __init__(self, in_feature, out_feature, hidden_feature, device):
        super().__init__()
        self.linear1 = Linear(in_feature, hidden_feature, device=device)
        self.linear2 = Linear(hidden_feature, out_feature, device=device)
        self.drop_out = Dropout(0.5)
        self.relu = ReLU()

    def forward(self, x):
        hidden = self.linear1(x)
        result = self.relu(hidden)
        result = self.drop_out(result)
        hidden = self.linear2(result)
        return hidden


def generate_list_data(files_list):
    to = []
    for path in files_list:
        with open(path, 'r', encoding='utf-8') as f:
            ids = json.load(f)['ids']
            for k in range(len(ids) - 1, -1, -1):
                if len(ids[k]) > 127:
                    del ids[k]
            max_length = (max(len(sublist) for sublist in ids) + 1)
            ids.insert(0, [0] * max_length)
            for i in range(len(ids)):
                ids[i] = torch.tensor(ids[i], dtype=torch.int16)
            to_return = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
            to.append((to_return, to_return != 0, path))
    return to


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts] for i in range(wanted_parts)]


def generate_steps(files_list):
    for path in files_list:
        with open(path, 'r', encoding='utf-8') as f:
            ids = json.load(f)['ids']
            for i in range(len(ids) - 1, -1, -1):
                if len(ids[i]) > 127 or len(ids[i]) < 10:
                    del ids[i]
            yield len(ids)


if __name__ == "__main__":
    _tokenizer = BertWordPieceTokenizer("custom/vocab.txt")
    # _tokenizer = BertWordPieceTokenizer("/mnt/c/Users/123/PycharmProjects/torch-models/custom/vocab.txt")
    char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)]
    char_list = char_list[:char_list.index('FT') + 1]
    # files = [fr"/mnt/c/Users/123/PycharmProjects/words2/{chars}/wiki_{i:02}.json_sentence.json_json.json.json" for i in
    #          range(15) for chars in char_list]
    files = [fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json_sentence.json_json.json.json" for i in
             range(15) for chars in char_list]
    count = 0
    _embedding_dim = 384
    _hidden_size = 3072
    _num_head = 12
    _out_dim = 512
    max_epoch = 5
    batch = 140
    _num_layers = 12
    vocab_size = _tokenizer.get_vocab_size()
    bert = Bert(_embedding_dim, _hidden_size, _num_head, 128, _num_layers, _tokenizer)
    bert.load_state_dict(torch.load("bert.pth"))
    bert.train()
    # layer = FeedForward2(384, 2, 3072, device=bert.device)
    layer = Linear(384, 2, device=bert.device)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(bert.parameters()), lr=1e-4)
    trainer = Train(bert, optimizer)
    trainer.add_bar('Epoch', 'Iter')
    trainer.add_metrics('loss', float)
    trainer.down_stream(batch, max_epoch, layer, log_dir="Bert_down_stream", log=True,
                        log_file_name="Bert_down_stream", monitor=False)
