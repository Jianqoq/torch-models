
import torch
from torch.nn import Module, MultiheadAttention, Linear, Embedding, Softmax, LayerNorm, ReLU, functional, CrossEntropyLoss
from layers import get_question_and_answer, Train
from torch import tensor
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transformer(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, _pad_index):
        super().__init__()
        self.encoder = Encoder(corpus, embedding_dim, hidden_size, num_head)
        self.decoder = Decoder(corpus, embedding_dim, hidden_size, num_head)
        self.linear = Linear(embedding_dim, corpus)
        self.softmax = Softmax(dim=2)
        self.loss = CrossEntropyLoss(ignore_index=_pad_index)
        self._pad_index = _pad_index
        self.num_heads = num_head
        self.offset = None

    def forward(self, questions, answers):
        decoder_input, decoder_target = answers[:, :-1], answers[:, 1:]
        # if self.offset is None:
        #     offset = questions.size(1) - answers.size(1)
        #     self.offset = offset
        #     mask = self.get_mask(questions, decoder_input)
        # else:
        #     mask = self.get_mask(questions, decoder_input)
        # mask = self.get_mask(questions, decoder_input)
        x = self.encoder(questions)
        new_sequence = self.decoder(decoder_input, x)
        score = self.linear(new_sequence)
        loss = self.loss.forward(torch.permute(score, (0, 2, 1)), decoder_target)
        return loss

    def generate(self, question, word_id, size):
        o2 = self.encoder.forward(question)
        sampled = []
        sample_id = tensor(word_id['_'], device=device).reshape(1, 1)
        total_hidden = o2
        for _ in range(size):
            x = sample_id.reshape((1, 1))
            new_sequence = self.decoder.forward(x, total_hidden)
            score = self.linear(new_sequence)
            score = self.softmax(score)
            sample_id = torch.argmax(score.flatten())
            sampled.append(sample_id)
        return sampled

    def get_mask(self, questions, answers):
        pad_index = self._pad_index
        offset = self.offset
        mask = None
        if offset > 0:
            answers = functional.pad(answers, (0, abs(offset)), value=pad_index)
            mask = answers.eq(pad_index)
        elif offset < 0:
            questions = functional.pad(questions, (0, abs(offset)), value=pad_index)
            mask = questions.eq(pad_index)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, len(answers[:, :-1][-1]))
            mask = torch.permute(mask.repeat(self.num_heads, 1, 1), (0, 2, 1))
        return mask


class Encoder(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.wordEmbedding = Embedding(corpus, embedding_dim)
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm2 = LayerNorm(embedding_dim)

    def forward(self, x):
        word_vec = self.wordEmbedding.forward(x)
        pe = positionalEncoding(word_vec)
        word_vec = word_vec + pe
        x, weights = self.multiHeadAttention(word_vec, word_vec, word_vec)
        o = self.layerNorm.forward(x + word_vec)
        ff_result = self.feedForward.forward(o)
        o2 = self.layerNorm2.forward(ff_result + o)
        return o2


class Decoder(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.wordEmbedding = Embedding(corpus, embedding_dim)
        self.maskedMultiHeadAttention = MultiheadAttention(embedding_dim, num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_head, batch_first=True)
        self.layerNorm2 = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm3 = LayerNorm(embedding_dim)
        self.num_heads = num_head

    def forward(self, answer, o2):
        word_vec = self.wordEmbedding.forward(answer)
        pe = positionalEncoding(word_vec)
        word_vec = word_vec + pe
        mask2 = torch.triu(torch.ones(word_vec.shape[0]*self.num_heads, len(answer[-1]), len(answer[-1]), device=device), diagonal=1).bool()
        mask = mask2
        # when doing query @ key, result is in shape (batch, seq_len, seq_len)
        # so mask has to in shape (batch, seq_len, seq_len)
        masked, weights = self.maskedMultiHeadAttention.forward(word_vec, word_vec, word_vec, attn_mask=mask)
        o = self.layerNorm.forward(masked + word_vec)
        x, weights = self.multiHeadAttention.forward(o, o2, o2)
        o2 = self.layerNorm2.forward(x + o)
        ff_result = self.feedForward.forward(o2)
        return self.layerNorm3.forward(ff_result + o2)

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        total_hidden = enc_hs
        for _ in range(sample_size):
            x = sample_id.reshape((1, 1))
            score = self.forward(x, total_hidden)
            sample_id = torch.argmax(score.flatten())
            sampled.append(sample_id)
        return sampled


class FeedForward(Module):
    def __init__(self, in_feature, out_feature, hidden_feature):
        super().__init__()
        self.linear1 = Linear(in_feature, hidden_feature)
        self.linear2 = Linear(hidden_feature, out_feature)
        self.relu = ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def positionalEncoding(x):  # x shape (batch, seq_len, embedding_dim)
    batch_size, seq_len, embedding_dim = x.shape
    i_mat = torch.pow(10000, torch.arange(0, embedding_dim, 2, device=device).float() / embedding_dim)
    pos = torch.arange(0, seq_len, device=device).unsqueeze(1).repeat(1, embedding_dim).float()
    pos = pos.unsqueeze(0).repeat(batch_size, 1, 1)
    temp_sine = pos[..., ::2]
    temp_cos = pos[..., 1::2]
    PE_sin = torch.sin(temp_sine / i_mat)
    PE_cos = torch.cos(temp_cos / i_mat)
    temp_sine[...] = PE_sin
    temp_cos[...] = PE_cos

    return pos


train_questions, test_questions, train_answer, test_answer, word_id, id_word = get_question_and_answer(
    r"C:\Users\123\PycharmProjects\torch-models\data-set\symbolic computation\multiplication_shuffle.txt",
    torch=True, train_ratio=0.96)
_pad_index = -1
_vocab_size = len(word_id)
_embedding_dim = 128
_hidden_size = 516
_num_head = 8
_out_dim = 512
max_epoch = 200
batch = 10
transformer = Transformer(_vocab_size, _embedding_dim, _hidden_size, _num_head, _pad_index)
transformer.to(device)
transformer.train()
optimizer = torch.optim.Adam(transformer.parameters())
trainer = Train(transformer, optimizer)
trainer.PYTORCH_train(train_questions, train_answer, test_questions, test_answer, batch,
                      max_epoch, word_id, id_word, log_dir="division", log=False,
                      log_file_name="Encoder1LSTM, Decoder1LSTM + 1Attention + affine")