import torch
from torch.nn import Module, MultiheadAttention, Linear, Embedding, Softmax, LayerNorm, ReLU, functional, CrossEntropyLoss
from layers import get_question_and_answer
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transformer(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.encoder = Encoder(corpus, embedding_dim, hidden_size, num_head)
        self.decoder = Decoder(corpus, embedding_dim, hidden_size, num_head)
        self.linear = Linear(embedding_dim, corpus, device=device)
        self.softmax = Softmax(dim=2)
        self.loss = CrossEntropyLoss()

    def forward(self, questions, answers):
        offset = questions.size(1) - answers.size(1)
        if offset > 0:
            answers = functional.pad(answers, (0, offset))
        elif offset < 0:
            questions = functional.pad(questions, (0, offset))
        decoder_input, decoder_target = answers[:, :-1], answers[:, 1:]
        x = self.encoder(questions)
        new_sequence = self.decoder(decoder_input, x)
        score = self.linear(new_sequence)
        possibilities = self.loss.forward(torch.permute(score, (0, 2, 1)), decoder_target)
        return possibilities


class Encoder(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.wordEmbedding = Embedding(corpus, embedding_dim, device=device)
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, device=device, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim, device=device)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm2 = LayerNorm(embedding_dim, device=device)

    def forward(self, x):
        word_vec = self.wordEmbedding.forward(x)
        x, weights = self.multiHeadAttention(word_vec, word_vec, word_vec)
        o = self.layerNorm.forward(x + word_vec)
        ff_result = self.feedForward.forward(o)
        o2 = self.layerNorm2.forward(ff_result + o)
        return o2


class Decoder(Module):
    def __init__(self, corpus, embedding_dim, hidden_size, num_head):
        super().__init__()
        self.wordEmbedding = Embedding(corpus, embedding_dim, device=device)
        self.maskedMultiHeadAttention = MultiheadAttention(embedding_dim, num_head, device=device, batch_first=True)
        self.layerNorm = LayerNorm(embedding_dim, device=device)
        self.multiHeadAttention = MultiheadAttention(embedding_dim, num_heads=num_head, device=device, batch_first=True)
        self.layerNorm2 = LayerNorm(embedding_dim, device=device)
        self.feedForward = FeedForward(embedding_dim, embedding_dim, hidden_size)
        self.layerNorm3 = LayerNorm(embedding_dim, device=device)

    def forward(self, answer, o2):
        word_vec = self.wordEmbedding.forward(answer)
        masked, weights = self.maskedMultiHeadAttention.forward(word_vec, word_vec, word_vec)
        o = self.layerNorm.forward(masked + word_vec)
        x, weights = self.multiHeadAttention.forward(o, o2, o2)
        o2 = self.layerNorm2.forward(x + o)
        ff_result = self.feedForward.forward(o2)
        return self.layerNorm3.forward(ff_result + o2)


class FeedForward(Module):
    def __init__(self, in_feature, out_feature, hidden_feature):
        super().__init__()
        self.linear1 = Linear(in_feature, hidden_feature, device=device)
        self.linear2 = Linear(hidden_feature, out_feature, device=device)
        self.relu = ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


train_questions, test_questions, train_answer, test_answer, word_id, id_word = get_question_and_answer(
    r"C:\Users\123\PycharmProjects\torch-models\data-set\symbolic computation\addition_shuffle2.txt",
    torch=True, train_ratio=0.96)
_vocab_size = len(word_id)
_embedding_dim = 128
_hidden_size = 516
_num_head = 8
_out_dim = 512
transformer = Transformer(_vocab_size, _embedding_dim, _hidden_size, _num_head)
transformer.train()
loss = transformer.forward(train_questions[:10], train_answer[:10])
loss.backward()
