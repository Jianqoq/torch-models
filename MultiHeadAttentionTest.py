
from torch import nn, tensor
import torch
from layers import get_question_and_answer, Train
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Seq2Seq(torch.nn.Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, num_layers):
        super().__init__()
        self.encoder = Encoder(corpus, embedding_dim, hidden_size, num_layers)
        self.decoder = Decoder(corpus, embedding_dim, hidden_size, num_head, num_layers)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        decoder_input, decoder_target = target[:, :-1], target[:, 1:]
        x, h_n = self.encoder(x)
        new_sequence = self.decoder(decoder_input, x, h_n)
        loss = self.loss(torch.permute(new_sequence, (0, 2, 1)), decoder_target)
        return loss

    def generate(self, question, word_id, size):
        x, h_n = self.encoder.forward(question)
        answer = self.decoder.generate(x, h_n, tensor(word_id['_'], device=device).reshape(1, 1), size)
        return answer


class Encoder(torch.nn.Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.word_embedding = nn.Embedding(corpus, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = self.word_embedding.forward(x)
        x, (h_n, c_n) = self.lstm.forward(x)
        return x, h_n


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_head, num_layers):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear3 = nn.Linear(hidden_size, vocab_size)
        self.multi_attention = nn.MultiheadAttention(hidden_size, num_head, batch_first=True)
        self.h_n, self.c_n = None, None
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

    def forward(self, answer, hidden_state, h_n, stateful=False):  # hidden_state (batch, h_t, hidden_size)
        h_n = self.h_n if self.h_n is not None else h_n
        c_n = self.c_n if self.c_n is not None else torch.zeros(h_n.shape, device=h_n.device)
        x = self.word_embedding.forward(answer)
        new_sequence, (h_n, c_n) = self.lstm.forward(x, (h_n, c_n))
        new_sequence, weights = self.multi_attention.forward(new_sequence, hidden_state, hidden_state)
        if stateful:
            self.h_n = h_n
            self.c_n = c_n
        else:
            self.h_n, self.c_n = None, None
        new_sequence = self.linear3.forward(new_sequence)
        return new_sequence

    def generate(self, enc_hs, h_n, start_id, sample_size):
        sampled = []
        sample_id = start_id
        total_hidden = enc_hs
        last_hidden = h_n
        for _ in range(sample_size):
            x = sample_id.reshape((1, 1))
            score = self.forward(x, total_hidden, last_hidden, stateful=True)
            sample_id = torch.argmax(score.flatten())
            sampled.append(sample_id)
        self.h_n, self.c_n = None, None
        self.h_n2, self.c_n2 = None, None
        return sampled


batch = 10
sentence_length = 30
corpus_size = 100
train_questions, test_questions, train_answer, test_answer, word_id, id_word = get_question_and_answer(
    r"C:\Users\123\PycharmProjects\torch-models\data-set\symbolic computation\multiplication_shuffle.txt",
    torch=True, train_ratio=0.96)
vocab_size = len(word_id)
wordvec_size = 16
hidden_size = 128
max_epoch = 50
num_head = 8
num_layers = 1
output_dim = 64
seq2seq = Seq2Seq(vocab_size, wordvec_size, hidden_size, num_head, num_layers)
seq2seq.to(device)
optimizer = torch.optim.Adam(seq2seq.parameters())
trainer = Train()
seq2seq.train()
trainer.PYTORCH_train(seq2seq, optimizer, train_questions, train_answer, test_questions, test_answer, batch,
                      max_epoch, word_id, id_word, log_dir="multiplication", log=True,
                      log_file_name="Encoder3LSTM, Decoder1LSTM + 1Attention + 2LSTM + affine")
