import torch

torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention:
    """
    simple version of multi-head attention
    """
    def __init__(self, hiddens, num_heads=8):
        self.Wq = torch.nn.Linear(hiddens, hiddens)
        self.Wk = torch.nn.Linear(hiddens, hiddens)
        self.Wv = torch.nn.Linear(hiddens, hiddens)
        self.Wo = torch.nn.Linear(hiddens, hiddens)
        self.num_heads = num_heads

    def forward(self, query, key, value):
        q = self.Wq(query)
        k = self.Wk(key)
        v = self.Wv(value)

        # split the tensor into num_heads
        """
        assume split into 3 heads
        a = array([[[ 0,  1,  2],
                    [ 3,  4,  5],
                    [ 6,  7,  8],
                    [ 9, 10, 11],
                    [12, 13, 14]],

                   [[15, 16, 17],
                    [18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26],
                    [27, 28, 29]]])
        a.reshape(a.shape[0], a.shape[1], 3, int(b.shape[2] / 3))
        a[:, :, 0, :] = array([[[ 0],
                                [ 3],
                                [ 6],
                                [ 9],
                                [12]],
                        
                               [[15],
                                [18],
                                [21],
                                [24],
                                [27]]])
        """
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, int(q.shape[2] / self.num_heads)).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, int(k.shape[2] / self.num_heads)).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, int(v.shape[2] / self.num_heads)).permute(0, 2, 1, 3)

        # transpose it so that we can calculate the dot product for all vectors
        """
        q = (batch, num_heads, sentence_length, each_head_hidden)
        transposed_shape = (batch, num_heads, each_head_hidden, sentence_length)
        score table = (batch, num_heads, sentence_length, sentence_length)
        Each element in the score table means the hidden weight relationship between the query and the key.
        """
        transposed = k.permute(0, 1, 3, 2)
        weight = q @ transposed

        scaled_score = weight / torch.sqrt(torch.tensor(int(v.shape[2] / self.num_heads)))

        """
        assume table is (5, 5), where first 10 means the word in the sentence, and the second 10 means the weight of
        the word in the sentence based on the first word.
                         Today  is   a   good   day
        table = Today   [[0.1, 0.2, 0.6, 0.05, 0.05]
                is       [0.05, 0.6, 0.2, 0.1, 0.05]
                a        [0.1, 0.6, 0.2, 0.05, 0.05]
                good     [0.1, 0.2, 0.1, 0.3,   0.3]
                day      [0.1, 0.2, 0.6, 0.05, 0.05]]
        We can see that the score table tells us that the word "Today" has a strong relationship with the word "a".
        """
        weight = torch.softmax(scaled_score, dim=-1)

        """
        now we get the possibility (batch, num_heads, sentence_length, possibility)
        where last two axes means there are sentence_length vectors with size possibility,
         and each vector has possibility.
        Since the hidden size is 512, which means we have 64 vectors for each head on the right operand.
        Since the score table is (10, 10), which means we have 10 vectors on the left operand.
        We use matmul to calculate the dot product for all vectors and finally we have 640 vectors for each head.
        Each element in the vector means the word the model is focusing on based on the weight.
        
        The Output is like extract the information from the sentence, and each word get a new hidden vector compare with
        the original hidden vector because we added small partial of hidden weights from other words.
        """
        output = torch.matmul(weight, v)

        # reshape back to the original shape
        merged = output.permute(0, 2, 1, 3).reshape(output.shape[0], output.shape[2], -1)
        return self.Wo(merged)


layer = MultiHeadAttention(512)
inp = torch.randn(64, 10, 512)
key = torch.randn(64, 10, 512)
value = torch.randn(64, 10, 512)
res = layer.forward(inp, key, value)
res.mean().backward()
assert res.shape == torch.Size([64, 10, 512])
