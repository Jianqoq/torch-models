
from torch import nn, tensor

import torch
from layers import get_question_and_answer, Train


class Transformer(torch.nn.Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, num_layers):
        super().__init__()
        self.encoder = Encoder(corpus, embedding_dim, hidden_size, num_head, num_layers)
        self.decoder = Decoder(corpus, embedding_dim, hidden_size, num_head, num_layers)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        decoder_input, decoder_target = target[:, :-1], target[:, 1:]
        x = self.encoder(x)
        new_sequence, h_n, c_n = self.decoder(decoder_input, x)
        loss = self.loss(torch.permute(new_sequence, (0, 2, 1)), decoder_target)
        return loss

    def generate(self, question, word_id, size):
        h = self.encoder.forward(question)
        answer = self.decoder.generate(h, tensor(word_id['_']).reshape(1, 1), size)
        return answer


class Encoder(torch.nn.Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, num_layers):
        super().__init__()
        self.word_embedding = nn.Embedding(corpus, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        # self.multi_attention = nn.MultiheadAttention(hidden_size, num_head, batch_first=True)
        # self.linear = nn.Linear(hidden_size, hidden_size)
        # self.multi_attention2 = nn.MultiheadAttention(hidden_size, num_head, batch_first=True)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.word_embedding.forward(x)
        x, (h_n, c_n) = self.lstm.forward(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_head, num_layers):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        # self.multi_attention = nn.MultiheadAttention(hidden_size, num_head, batch_first=True)
        # self.linear = nn.Linear(hidden_size, hidden_size)
        # self.multi_attention2 = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.multi_attention3 = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
        self.linear3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, answer, hidden_state, c_t=None):  # hidden_state (h_t, batch, hidden_size)
        h_n = hidden_state[:,-1].reshape(1, hidden_state[:,-1].shape[0], hidden_state[:,-1].shape[1])
        x = self.word_embedding.forward(answer)

        x, (h_n, c_n) = self.lstm.forward(x, (h_n, torch.zeros(h_n.shape))) if c_t is None else self.lstm.forward(x, (h_n, c_t))
        new_sequence = self.linear3.forward(x)
        return new_sequence, h_n, c_n

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        c_t = None
        h_n = enc_hs
        for _ in range(sample_size):
            x = tensor([sample_id]).reshape((1, 1))
            score, h_n, c_t = self.forward(x, h_n, c_t)
            sample_id = torch.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


batch = 10
sentence_length = 30
corpus_size = 100
train_questions, test_questions, train_answer, test_answer, word_id, id_word = get_question_and_answer("addition_shuffle2.txt")
test_questions = tensor(test_questions)
test_answer = tensor(test_answer)
train_questions = tensor(train_questions)
train_answer = tensor(train_answer).long()
vocab_size = len(word_id)
wordvec_size = 16
hidden_size = 128
max_epoch = 200
num_head = 8
num_layers = 1
output_dim = 64
transformer = Transformer(vocab_size, wordvec_size, hidden_size, num_head, num_layers)
optimizer = torch.optim.Adam(transformer.parameters())
trainer = Train()
transformer.train()
trainer.PYTORCH_train(transformer, optimizer, train_questions, train_answer, test_questions, test_answer,
                    batch, max_epoch, word_id, id_word, "layers/runs", Enale_Tensorboard=False)
