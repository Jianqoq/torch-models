
import torch
from torch.nn import Module, MultiheadAttention, Linear, Embedding, Softmax, LayerNorm, ReLU, CrossEntropyLoss, ModuleList
from layers import get_question_and_answer, Train, Preprocess
from torch import tensor
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transformer(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head, _pad_index, num_layers):
        super().__init__()
        self.wordEmbedding1 = Embedding(corpus, embedding_dim)
        self.encoder = ModuleList([Encoder(embedding_dim, hidden_size, num_head) for _ in range(num_layers)])

        self.wordEmbedding2 = Embedding(corpus, embedding_dim)
        self.decoder = ModuleList([Decoder(embedding_dim, hidden_size, num_head) for _ in range(num_layers)])
        self.linear = Linear(embedding_dim, corpus)
        self.loss = CrossEntropyLoss(ignore_index=_pad_index)
        self._pad_index = _pad_index
        self.num_heads = num_head
        self.offset = None

    def forward(self, questions, answers):
        decoder_input, decoder_target = answers[:, :-1], answers[:, 1:]
        word_vec = self.wordEmbedding1.forward(questions)
        pe = positionalEncoding(word_vec)
        encoder_output = word_vec + pe
        for i in self.encoder:
            encoder_output = i(encoder_output)

        word_vec = self.wordEmbedding2.forward(decoder_input)
        pe = positionalEncoding(word_vec)
        word_vec = word_vec + pe
        for i in self.decoder:
            word_vec = i(decoder_input, word_vec, encoder_output)

        score = self.linear(word_vec)
        loss = self.loss.forward(torch.permute(score, (0, 2, 1)), decoder_target)
        return loss

    def generate(self, question, word_id, size):
        word_vec = self.wordEmbedding1.forward(question)
        pe = positionalEncoding(word_vec)
        encoder_output = word_vec + pe
        for i in self.encoder:
            encoder_output = i(encoder_output)

        sampled = []
        sample_id = word_id['_']
        sampled.append(sample_id)
        total_hidden = encoder_output

        for _ in range(size):
            x = tensor(sampled, device=device).reshape((1, len(sampled)))
            word_vec = self.wordEmbedding2.forward(x)
            pe = positionalEncoding(word_vec)
            word_vec = word_vec + pe
            for i in self.decoder:
                word_vec = i(x, word_vec, total_hidden)
            score = self.linear(word_vec)
            score = self.softmax(score)[:, -1, :]
            sample_id = int(torch.argmax(score.flatten()))
            sampled.append(sample_id)
        return sampled[1:]


class Encoder(Module):
    def __init__(self, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm2 = LayerNorm(embedding_dim)

    def forward(self, word_vec):
        x, weights = self.multiHeadAttention(word_vec, word_vec, word_vec)
        o = self.layerNorm.forward(x + word_vec)
        ff_result = self.feedForward.forward(o)
        o2 = self.layerNorm2.forward(ff_result + o)
        return o2


class Decoder(Module):
    def __init__(self, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.maskedMultiHeadAttention = MultiheadAttention(embedding_dim, num_head, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim)
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_head, batch_first=True)
        self.layerNorm2 = LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm3 = LayerNorm(embedding_dim)
        self.num_heads = num_head

    def forward(self, answer, word_vec, o2):
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


# with open("data-set/corpus/ptb.train.txt") as fp:
#     string = fp.readlines()
# preprocess = Preprocess(string, ' ', '.')
train_questions, test_questions, train_answer, test_answer, word_id, id_word = get_question_and_answer(
    r"C:\Users\123\PycharmProjects\torch-models\data-set\symbolic computation\addition_shuffle2.txt",
    torch=True, train_ratio=0.96)
_pad_index = -1
_vocab_size = len(word_id)
_embedding_dim = 128
_hidden_size = 516
_num_head = 8
_out_dim = 512
max_epoch = 50
batch = 10
_num_layers = 2
transformer = Transformer(_vocab_size, _embedding_dim, _hidden_size, _num_head, _pad_index, _num_layers)
transformer.to(device)
transformer.train()
optimizer = torch.optim.Adam(transformer.parameters())
trainer = Train(transformer, optimizer)
trainer.PYTORCH_train(train_questions, train_answer, test_questions, test_answer, batch,
                      max_epoch, word_id, id_word, log_dir="runs", log=True,
                      log_file_name="Transformer")
