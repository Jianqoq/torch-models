import math
import os
import sys

import ad
import time
import random
import numpy as np
from sklearn.base import BaseEstimator
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)


def softmax(x, dims=0):
    x = x - x.max(axis=dims, keepdims=True)
    x_exp = np.exp(x)
    result = np.sum(x_exp, axis=dims, keepdims=True)
    x = x_exp / result
    return x


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break
        if not find_flg:
            break

    return params, grads


class Repeat:
    def __init__(self, offset_var, repeat_num, axis: str):
        axis = axis.lower()
        if axis == 'y':
            self.repeated = np.repeat(offset_var, repeat_num, axis=0)
        elif axis == 'x':
            self.repeated = np.repeat(offset_var, repeat_num, axis=1)
        else:
            raise ValueError('axis should be x or y')


class Sum:
    def forward(self, x) -> np.ndarray:
        return np.sum(x, axis=0, keepdims=True)

    def backward(self, dy, collums) -> np.ndarray:
        return np.repeat(dy, collums, axis=0)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


class RNN:
    """
    :param w_input: shape(vocab_size, hidden_size)
    :param w_prev: shape(hidden_size, hidden_size)
    """

    def __init__(self, w_input, w_prev, b):
        self.weights, self.grads = [w_input, w_prev, b], []
        self.Matmul1 = MatMul(w_prev)
        self.Matmul2 = MatMul(w_input)
        self.cache = None

    def forward(self, word_vector, h_prev):
        """
        :param word_vector: shape(mini_batch_size, word_vector)
        :param h_prev: shape(mini_batch_size, hidden_size)

        new_h = shape(mini_batch_size, hidden_size)
        new_x = shape(mini_batch_size,

        :return:
        """

        w_input, w_prev, b = self.weights
        new_h = self.Matmul1.forward(h_prev)
        new_x = self.Matmul2.forward(word_vector)
        first_total = new_x + new_h
        # repeat = Repeat()
        final = first_total + b
        h_next = np.tanh(final)
        self.cache = (word_vector, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        dt = dh_next * (1 - np.square(h_next))
        db = np.sum(dt, axis=0)
        dx = self.Matmul1.backward(db)
        dh_next = self.Matmul2.backward(db)
        return dh_next, dx


class MatMul:

    def __init__(self, W):
        self.weights = [W]
        self.X = None
        self.gradients = [np.zeros_like(W)]

    def forward(self, forward_input):
        W, = self.weights
        output = np.dot(forward_input, W)
        self.X = forward_input
        return output

    def backward(self, d_backward_input):
        # get weights and calculate dX
        W = self.weights[0]
        dX = np.dot(d_backward_input, W.T)

        # use stored input to and dinput to calculate dW and store to self.gradients list
        dW = np.dot(self.X.T, d_backward_input)
        self.gradients[0][...] = dW

        return dX


class TimeRNN:
    def __init__(self, w_input, w_prev, b, stateful=False):
        self.weights = [w_input, w_prev, b]
        self.grads = [np.zeros_like(w_input), np.zeros_like(w_prev), np.zeros_like(b)]
        self.stateful = stateful
        self.layers = None
        self.h, self.dh = None, None  # self.h = RNN output

    def forward(self, x_sequence):
        """
        n: batch size
        """
        w_cols, hidden_size = self.weights[0].shape
        mini_batch, number_words, w_cols = x_sequence.shape
        hs = np.empty((mini_batch, number_words, hidden_size), dtype='f')
        self.layers = []
        if not self.stateful or self.h is None:
            self.h = np.zeros((mini_batch, hidden_size), dtype='f')  # not store previous RNN output
        for i in range(number_words):
            layer = RNN(*self.weights)
            self.h = layer.forward(x_sequence[:, i, :], self.h)  # x_sequence[:, i, :] = word vector
            self.layers.append(layer)
            hs[:, i, :] = self.h
        return hs

    def backward(self, dhs):
        wx, wh, b = self.weights
        n, t, d = dhs.shape
        dxs = np.empty((n, t, wx.shape[0]), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for i in reversed(range(t)):
            layer = self.layers[i]
            dh, dx = layer.backward(dhs[:, i, :] + dh)
            dxs[:, i, :] = dx
            for index, k in enumerate(layer.grads):
                grads[index] += k
        for i, grad in grads:
            self.grads[i][...] = grad
        self.dh = dh
        return dxs


class SoftmaxWithLoss:
    """
    general SoftmaxWithLoss
    """

    def __init__(self):
        self.x, self.y = None, None

    def forward(self, x, true):
        score = softmax(x)

        def multi_cross_entropy_error(y, t):
            if y.ndim == 1:
                t = t.reshape(1, t.size)
                y = y.reshape(1, y.size)

            if t.size == y.size:
                t = t.argmax(axis=1)

            batch_size = y.shape[0]
            return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        loss = multi_cross_entropy_error(score, true)
        self.y = score
        self.x = true
        return loss

    def backward(self):
        return self.y - self.x


class TimeSigmoidWithLoss:
    def __init__(self, length):
        self.layers = []
        for i in range(length):
            self.layers.append(SoftmaxWithLoss())

    def forward(self, x_sequence, t_sequence):
        total_loss = 0
        for index, i in enumerate(zip(x_sequence, t_sequence)):
            total_loss += self.layers[index].forward(*i)
        return total_loss / x_sequence.shape[1]

    def backward(self):
        return np.array([i.backward() for i in self.layers])


class TimeAffine:
    def __init__(self, w, b):
        self.weights = [w, b]
        self.x = None
        self.forward_shape = None
        self.grads = [np.zeros_like(w), np.zeros_like(b)]

    def forward(self, x_sequence):
        """
        :param x_sequence: same as the input of x_sequence in TimeRNN Layer

        x = np.array([[[1,2,3],[3,4,5]],[[5,6,7],[7,8,9]]])
        x = [[[1 2 3]   x.shape = (2, 2, 3)
            [3 4 5]]
            [[5 6 7]
            [7 8 9]]]

        y = np.array([[1, 2, 3, 1, 1, 1],[3, 4, 5, 1, 1, 1],[5, 6, 7, 1, 1, 1]])
        y = [[1 2 3 1 1 1]  y.shape = (3, 6)
            [3 4 5 1 1 1]
            [5 6 7 1 1 1]]

        np.dot(x, y) = [[[ 22,  28,  34,   6,   6,   6],
                       [ 40,  52,  64,  12,  12,  12]],
                       [[ 58,  76,  94,  18,  18,  18],
                       [ 76, 100, 124,  24,  24,  24]]]
        y[:,0] = [1, 3, 5]
        x[0,0,:] = [1, 2, 3]
        sum(x[0,0,:]*y[:,0]) = 22
        sum(x[0,0,:]*y[:,1]) = 28

        """
        N, T, D = x_sequence.shape
        W, b = self.weights

        rx = x_sequence.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x_sequence
        out = out.reshape(N, T, -1)
        self.forward_shape = out.shape
        return out
        # efficient way
        # w, b = self.weights
        # self.x = x_sequence
        # out = np.dot(x_sequence, w) + b
        # return out
        # easy way and understandable way
        # vocab_size = self.weights[0].shape[1]
        # mini_batch, number_words, hidden_size = x_sequence.shape
        # out = np.empty((mini_batch, number_words, vocab_size), dtype='f')
        # for index, i in enumerate(self.layers):
        #     out[:, index, :] = i.forward(x_sequence[:, index, :])
        # return out

    def backward(self, d_sequence):
        x = self.x
        N, T, D = x.shape
        W, b = self.weights

        dout = d_sequence.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx
        # for i in self.layers:
        #     i.backward(d_sequence)


class Preprocess:
    def __init__(self, text: str, split_way=' ', *args):
        dictionary = {i: f' {i}' for i in args}
        text = text.lower()
        for i in dictionary:
            text = text.replace(i, dictionary.get(i))
        self.text = text.split(split_way)

    @staticmethod
    def get_word_id(text):
        """has to be unique for each element"""
        word_id = {}
        id_word = {}
        corpus = []
        append = corpus.append
        counter = 3
        for index, i in enumerate(text):
            word_id[i] = counter
            id_word[counter] = i
            counter += 1
            append(word_id[i])
        return word_id, id_word, corpus

    def get_single_context(self, id_word: dict, word_id: dict, corpus: list, word: str,
                           window: int):  # list bound check
        text = self.text
        word = word.lower()
        length = len(text)
        if word not in text:
            return
        ls = [0] * len(corpus)
        for index, i in enumerate(text):
            if word_id[i] == word_id[word]:
                if index == 0:
                    counter = 1
                    for k in range(window):
                        ls[counter] += 1
                        counter += 1
                elif index == length - 1:
                    counter = 1
                    for p in range(window):
                        ls[-1 - counter] += 1
                        counter += 1
                else:
                    counter = counter2 = 1
                    word1_id = word_id[text[index - counter]]
                    word2_id = word_id[text[index + counter2]]
                    for p in range(window):
                        ls[word1_id] += 1
                        ls[word2_id] += 1
                        counter += 1
                        counter2 += 1

        return np.array(ls, dtype='uint8')

    def get_coocurrenceMatrix(self, corpus: list, id_word: dict, word_id: dict, window: int):
        ls = []
        append = ls.append
        total = len(word_id)
        begin = time()
        for index, i in enumerate(word_id):
            append(self.get_single_context(id_word, word_id, corpus, i, window))
            print_result(index + 1, total, begin, time())
        return np.array(ls, dtype='uint8'), ls

    def create_context_target(self, corpus, windowsize=1):
        target = corpus[1: -1]
        context = []
        cs = []
        for i in range(windowsize, len(corpus) - 1):
            cs.append(corpus[i - 1])
            cs.append(corpus[i + 1])
            context.append(cs)
            cs = []
        return np.array(context), np.array(target)

    def convert_onehot(self, context, target, length):
        zero_context = np.zeros(shape=(*context.shape, length), dtype='uint8')
        zero_target = np.zeros(shape=(*target.shape, length), dtype='uint8')
        for index, i in enumerate(context):
            for index2, k in enumerate(i):
                zero_context[index, index2, k] = 1
        for index, i in enumerate(target):
            zero_target[index, i] = 1
        return zero_context, zero_target

    def PPMI(self, co_matrix, corpus, verbose=True):
        ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
        N = np.sum(co_matrix)
        sigle_word = np.sum(co_matrix, axis=0)
        total = co_matrix.shape[0] * co_matrix.shape[1]
        cols = co_matrix.shape[1]
        cnt = 0
        begin = time.time()
        for i in range(co_matrix.shape[0]):
            for j in range(co_matrix.shape[1]):
                ppmi = np.log2(co_matrix[i, j] * N / (sigle_word[i] * sigle_word[j]) + 1e-8)
                ppmi_matrix[i, j] = max(0, ppmi)
                if verbose:
                    cnt += 1
                    if cnt % (total // 200) == 0:
                        print_result(cnt + 1, total, begin, time())
        return ppmi_matrix

    #
    # def most_similar(self, matrix: list, word: str, word_id: dict, top: int):
    #     word = word.lower()
    #     if word not in word_id:
    #         return
    #     word_use_vector = matrix[word_id[word]]
    #     ls = {id_word[index]: similarity(word_use_vector, i) for index, i in enumerate(matrix) if
    #           index is not word_id[word]}
    #     return sorted(ls.items(), key=lambda x: x[1], reverse=True)[:top]

    def similarity(self, vect1, vect2):
        x = vect1 / (np.sqrt(np.sum(vect1 ** 2)) + 1e-8)
        y = vect2 / (np.sqrt(np.sum(vect2 ** 2)) + 1e-8)
        return np.dot(x, y)


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class TimeEmbedding:
    def __init__(self, w):
        self.weights = [w]
        self.grads = [np.zeros_like(w)]
        self.layers = []
        # self.layers = []
        # self.weights = [w]
        # self.w = w
        # self.layers = [WordEmbed(w) for _ in range(length)]
        # self.grads = [np.zeros_like(w)]

    # def forward(self, x_sequence) -> np.ndarray:
    #     """
    #     word_vector = shape(1, w.shape[1])
    #     :param x_sequence: np.array([[1, 2, 3], [4, 5, 6]]) shape(2, 3) | 2 = mini_batch, 3 = number of words
    #     :return: np.ndarray
    #     """
    #     w = self.w
    #     mini_batch, number_words = x_sequence.shape
    #     w_rows, w_cols = w.shape
    #     out = np.empty((mini_batch, number_words, w_cols), dtype='f')
    #     for index, i in enumerate(self.layers):
    #         p = i.forward(x_sequence[:, index])
    #         out[:, index, :] = p
    #     return out
    def forward(self, xs):
        N, T = xs.shape
        V, D = self.weights[0].shape

        out = np.empty((N, T, D), dtype='f')
        self.layers.clear()

        for t in range(T):
            layer = Embedding(self.weights[0])
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, d_sequence):
        N, T, H = d_sequence.shape
        grad = 0
        for i in range(T):
            layer = self.layers[i]
            layer.backward(d_sequence[:, i, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.weights, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = ad.SoftmaxWithLoss(xs, 2).val
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx


class TimeLSTM:
    def __init__(self, wx, wh, b, stateful=False):
        self.weights = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.layers = []
        self.h, self.dh, self.c = None, None, None
        self.stateful = stateful
        self.forward_shape = None

    def forward(self, xs):
        wx, wh, b = self.weights
        batch, sentence_length, hidden_size_of_word_embedding = xs.shape
        hidden_size_of_lstm = wh.shape[0]

        hs = np.empty((batch, sentence_length, hidden_size_of_lstm), dtype='f')
        self.layers.clear()
        if not self.stateful or self.h is None:
            self.h = np.zeros((batch, hidden_size_of_lstm), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((batch, hidden_size_of_lstm), dtype='f')
        # loop word by word
        for i in range(sentence_length):
            layer = LSTM(*self.weights)
            self.c, self.h = layer.forward(xs[:, i, :], self.h, self.c)
            hs[:, i, :] = self.h
            self.layers.append(layer)
        self.forward_shape = hs.shape
        return hs

    def backward(self, dhs):
        wx, wh, b = self.weights
        batch, length, H = dhs.shape
        D = wx.shape[0]
        dxs = np.empty((batch, length, D), dtype='f')
        dh, dc = 0, np.zeros((batch, H), dtype='f')
        grads = [0, 0, 0]
        for i in reversed(range(length)):
            layer = self.layers[i]
            dx, dh, dc = layer.backward(dhs[:, i, :] + dh, dc)
            dxs[:, i, :] = dx
            for index, grad in enumerate(layer.grads):
                grads[index] += grad

        for index, grad in enumerate(grads):
            self.grads[index][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


def cache_graph(*args):
    return {id(i): i for i in args}


def update_cache(dictionary, *args):
    for idx, i in enumerate(dictionary):
        dictionary[i] = args[idx]


class LSTM:
    def __init__(self, wx, wh, b, dtype=np.float32):
        self.weights = [wx, wh, b]
        self.grads = [np.zeros_like(wx, dtype=dtype),
                      np.zeros_like(wh, dtype=dtype),
                      np.zeros_like(b, dtype=dtype)]
        self.__var = None

    def forward(self, x, h, c):
        """
        :return: shape (batch size, sentence length, hidden size of lstm layer)
        """
        wx, wh, b = self.weights

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        first = x @ wx + h @ wh + b

        slice1, slice2, slice3, slice4 = np.array([np.squeeze(i) for i in np.split(first, 4, axis=1)])

        slice1 = sigmoid(slice1)
        slice2 = np.tanh(slice2)
        slice3 = sigmoid(slice3)
        slice4 = sigmoid(slice4)

        ct = slice1 * c + slice2 * slice3
        ht = slice4 * np.tanh(ct)

        self.__var = (ct, ht, h, c, x, wh, wx, slice1, slice2, slice3, slice4)

        return ct, ht

    def backward(self, dh_next, dc_next):
        ct, ht, h, c, x, wh, wx, f, g, i, o = self.__var

        tang_c_2 = np.tanh(ct) ** 2
        dh_next_o = dh_next * o
        dh_next_o_i = dh_next_o * i
        dc_next_i = dc_next * i
        dh_next_o_f = dh_next_o * f
        tang_c_2_1 = 1 - tang_c_2
        f_1 = 1 - f
        i_1 = 1 - i

        stack0 = (dh_next_o_f * tang_c_2_1 + dc_next * f) * c * f_1
        stack1 = (dh_next_o_i * (1 - tang_c_2) + dc_next_i) * (1 - g ** 2)
        stack2 = (dh_next_o_i * tang_c_2_1 + dc_next_i) * g * i_1
        stack3 = dh_next * ht * (1 - o)
        db = np.hstack((stack0, stack1, stack2, stack3))

        dWx = x.T @ db
        dWh = h.T @ db
        dx = db @ wx.T
        dh = db @ wh.T
        dc = dh_next_o_f * (1 - tang_c_2) + dc_next * f

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = np.sum(db, axis=0)

        return dx, dh, dc


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.weights, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class Rnnlm:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初始化权重
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_W = embed_W.T
        affine_b = np.zeros(V).astype('f')

        # 生成层
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeDropout(),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = [self.layers[2], self.layers[4]]

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.weights
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for i in self.lstm_layer:
            i.reset_state()


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        total = max_epoch * max_iters
        begin = time.time()
        count = 0
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                count += 1
                ppl = np.exp(total_loss / loss_count)
                print_result(count, total, begin, ppl, optimizer.lr)
                # パープレキシティの評価
                if (eval_interval is not None) and not iters % eval_interval:
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def save(self):
        params, grads = remove_duplicate(self.model.params, self.model.grads)
        for index, i in enumerate(params):
            np.save(f"weights{index}.npy", i)


class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Encoder:
    def __init__(self, vocab_size, wordvec_size=100, hidden_size=100):
        if isinstance(vocab_size, list):
            vocab_size = vocab_size[0]
        w1, w2, w3, b = np.random.randn(vocab_size, wordvec_size) / 100, \
                        np.random.randn(wordvec_size, 4 * hidden_size) / np.sqrt(wordvec_size), \
                        np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size), \
            np.zeros(4 * hidden_size)
        self.layers = [TimeEmbedding(w1),
                       TimeLSTM(w2, w3, b, stateful=False)]

        self.weights = self.layers[0].weights + self.layers[1].weights
        self.grads = self.layers[0].grads + self.layers[1].grads
        self.hs = None

    def forward(self, xs):
        wordembed, lstm = self.layers
        out = wordembed.forward(xs)
        hs = lstm.forward(out)
        self.hs = hs
        # hs is a set of hidden state based on a sentence, we only need the last hidden state
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        wordembed, lstm = self.layers
        dout = lstm.backward(dhs)
        wordembed.backward(dout)


class Decoder:
    def __init__(self, vocab_size=12, wordvec_size=100, hidden_size=100):
        w1, w2, w3, w4, b = np.random.randn(vocab_size, wordvec_size) / 100, \
                            np.random.randn(wordvec_size + hidden_size, 4 * hidden_size) / np.sqrt(wordvec_size), \
                            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size), \
                            np.random.randn(hidden_size * 2, vocab_size) / np.sqrt(hidden_size), \
            np.zeros(4 * hidden_size)
        b1 = np.zeros(vocab_size)
        self.layers = [TimeEmbedding(w1),
                       TimeLSTM(w2, w3, b, stateful=True),
                       TimeAffine(w4, b1)]

        self.loss_layer = [TimeSoftmaxWithLoss()]
        self.weights = []
        self.grads = []
        self.hidden_size = hidden_size
        for layer in self.layers:
            self.weights += layer.weights
            self.grads += layer.grads
        self.cache = None
        self.hs = None

    def forward(self, xs, hidden_state):
        self.layers[1].set_state(hidden_state)
        time_embedding, time_lstm, time_affine = self.layers
        out = time_embedding.forward(xs)
        N, T = xs.shape
        hidden = np.repeat(hidden_state, T, axis=0).reshape(N, T, self.hidden_size)
        concate = np.concatenate((hidden, out), axis=-1)
        out = time_lstm.forward(concate)
        out = np.concatenate((hidden, out), axis=-1)
        out = time_affine.forward(out)
        return out

    def backward(self, dout):
        time_embedding, time_lstm, time_affine = self.layers
        dout = time_affine.backward(dout)
        dout, dhs0 = dout[:, :, self.hidden_size:], dout[:, :, :self.hidden_size]
        dout = time_lstm.backward(dout)
        dout, dhs1 = dout[:, :, self.hidden_size:], dout[:, :, :self.hidden_size]
        time_embedding.backward(dout)
        dhs = dhs0 + dhs1
        dh = time_lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, xs, size):
        sampled = []
        id = xs
        N, T = xs.shape
        embed, lstm, affine = self.layers
        lstm.set_state(h)

        hidden = h.reshape(N, T, self.hidden_size)
        for _ in range(size):
            out = embed.forward(id)

            concate = np.concatenate((hidden, out), axis=-1)
            out = lstm.forward(concate)
            out = np.concatenate((hidden, out), axis=-1)
            score = affine.forward(out)

            id = np.argmax(score, axis=2)
            sampled.append(int(id))

        return sampled


class Seq2Seq:
    def __init__(self, word_id, vocab_size, wordvec_size=100, hidden_size=100):
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.loss_layer = TimeSoftmaxWithLoss()
        self.word_id = word_id
        self.weights = self.encoder.weights + self.decoder.weights
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_input, decoder_target = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_input, h)
        loss = self.loss_layer.forward(score, decoder_target)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dh = self.decoder.backward(dout)
        self.encoder.backward(dh)
        pass

    def generate(self, question, word_id, size):
        h = self.encoder.forward(question)
        answer = self.decoder.generate(h, np.array(word_id['_']).reshape(1, 1), size)
        return answer

    def reset_state(self):
        self.decoder.layers[1].reset_state()


class WeightSum:
    def __init__(self):
        self.weights, self.grads = [], []
        self.softmax = None
        self.cache = None

    def forward(self, hs, a):
        # hidden_size == hidden_size
        batch, sentence_length, hidden_size = hs.shape
        # batch, sentence_length = a.shape

        ar = np.reshape(a, (batch, sentence_length, 1))
        t = ar * hs
        c = np.sum(t, axis=1)
        self.cache = (hs, ar, c, a)

        return c

    def backward(self, dout):
        hs, ar, c, a = self.cache
        batch, sentence_length, hidden_size = hs.shape
        dout = np.reshape(dout, (batch, 1, hidden_size))
        da = np.reshape(np.sum(dout * hs, axis=2, keepdims=True), (batch, sentence_length))
        dhs = np.repeat(dout, repeats=sentence_length, axis=1) * ar

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.weights, self.grads = [], []
        self.softmax = None
        self.cache = None

    def forward(self, hs, h):
        batch, sentence_length, hidden_size = hs.shape

        ar = np.reshape(h, (batch, 1, hidden_size))
        ar = np.repeat(ar, sentence_length, 1)
        t = ar * hs
        c = np.sum(t, axis=2)
        weight = softmax(c, c.ndim - 1)
        self.cache = (hs, ar, weight, h)
        return weight

    def backward(self, dout):
        hs, ar, weight, h = self.cache
        batch, sentence_length, hidden_size = hs.shape

        # 提取重复计算到变量
        reshaped_h = np.reshape(h, (batch, 1, hidden_size))
        repeated_h = np.repeat(reshaped_h, repeats=sentence_length, axis=1)

        # 计算公共子表达式
        weighted_sum = np.sum(repeated_h * hs, axis=2)
        softmax_weighted_sum = softmax(weighted_sum, dims=1)
        sum_dout = np.sum(softmax_weighted_sum * dout, axis=1, keepdims=True)
        common_term = softmax_weighted_sum * dout - softmax_weighted_sum * sum_dout
        common_term_expanded = np.repeat(common_term[..., np.newaxis], hidden_size, axis=2)

        # 计算 hs1
        dh = np.reshape(np.sum(common_term_expanded * hs, axis=1, keepdims=True), (batch, hidden_size))

        # 计算 dhs1
        dhs = common_term_expanded * repeated_h

        return dhs, dh


class Attention:
    def __init__(self):
        self.weights, self.grads = [], []
        self.layers = [AttentionWeight(),
                       WeightSum()]
        self.hs_weight = None

    def forward(self, hs, h):
        attention_weight, weight_sum = self.layers
        a = attention_weight.forward(hs, h)
        c = weight_sum.forward(hs, a)
        self.hs_weight = a
        return c

    def backward(self, dout):
        attention_weight, weight_sum = self.layers
        hs0, dout = weight_sum.backward(dout)
        hs1, dout = attention_weight.backward(dout)
        hs = hs0 + hs1
        return hs, dout


class TimeAttention:
    def __init__(self):
        self.weights, self.grads = [], []
        self.layers = []
        self.hs_weight = []

    def forward(self, hs_enc, hs_dec):
        batch, sentence_length, hidden_size = hs_dec.shape
        out = np.empty(hs_dec.shape)
        self.layers.clear()
        self.hs_weight = []
        for i in range(sentence_length):
            t = Attention()
            out[:, i, :] = t.forward(hs_enc, hs_dec[:, i, :])
            self.layers.append(t)
            self.hs_weight.append(t.hs_weight)
        return out

    def backward(self, dout):
        batch, sentence_length, hidden_size = dout.shape
        dhs_dec = np.empty(dout.shape)
        dhs_enc = 0

        for i in range(sentence_length):
            layer = self.layers[i]
            dhs, dh = layer.backward(dout[:, i, :])
            dhs_enc += dhs
            dhs_dec[:, i, :] = dh
        return dhs_enc, dhs_dec


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        if isinstance(vocab_size, list):
            vocab_size = vocab_size[0]
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.weights, self.grads = [], []
        for layer in layers:
            self.weights += layer.weights
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.layers[0].forward(xs)
        hs = self.layers[1].forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.layers[1].backward(dhs)
        dhs = self.layers[0].backward(dout)
        return dhs


class AttentionSeq2seq(Seq2Seq, BaseEstimator):
    def __init__(self, word_id, vocab_size, wordvec_size, hidden_size):
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, wordvec_size, hidden_size)
        self.loss_layer = TimeSoftmaxWithLoss()
        self.weights = self.encoder.weights + self.decoder.weights
        self.grads = self.encoder.grads + self.decoder.grads
        self.word_id = word_id
        self.vocab_size = vocab_size
        self.wordvec_size = wordvec_size
        self.hidden_size = hidden_size

    def fit(self, train_questions, train_answer, **fit_params):
        optimizer = fit_params["optimizer"]
        max_epoch = fit_params["max_epoch"]
        batch_size = fit_params["batch_size"]
        max_iter = len(train_questions) // batch_size
        loss_cumulate = 0
        begin = time.time()
        global round0
        round0 += 1

        def clip_grad(grads, max_grad):
            total = 0
            for grad in grads:
                total += np.sum(grad ** 2)  # avoid broadcast
            norm = np.sqrt(total)
            if norm > max_grad:
                for grad in grads:
                    grad *= max_grad / norm

        for epoch in range(max_epoch):
            for iters in range(max_iter):
                batch_question = train_questions[iters * batch_size:(iters + 1) * batch_size]
                batch_answer = train_answer[iters * batch_size:(iters + 1) * batch_size]
                loss = self.forward(batch_question, batch_answer)
                self.backward()
                clip_grad(self.grads, fit_params["max_grad"])
                optimizer.update(self.weights, self.grads)

                loss_cumulate += loss

                if (iters + 1) % 10 == 0:
                    average_loss = loss_cumulate / 10
                    loss_cumulate = 0
                    print_result(epoch + 1, max_epoch, iters + 1, max_iter, begin, average_loss, optimizer.lr, round0)
        print('\n')

    def score(self, x, y):
        correct = 0
        size = len(y[0, :]) - 1

        for i in range(len(x)):
            question, answer = x[[i]], y[[i]]
            correct += evaluate(self, question, answer, self.word_id, size)

        return correct / len(x)

    def get_params(self, deep=False):
        return {
            'word_id': self.word_id,
            'vocab_size': self.vocab_size,
            'wordvec_size': self.wordvec_size,
            'hidden_size': self.hidden_size
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class FeedForward:
    def __init__(self, w, b, dtype=np.float32):
        self.weights = [w, b]
        self.grads = [np.zeros_like(w, dtype=dtype),
                      np.zeros_like(b, dtype=dtype)]
        self.cache = None

    def forward(self, x):
        w, b = self.weights

        z = x @ w + b

        relu = ReLU()
        self.cache = (w, x, b, relu)
        return relu.forward(z)

    def backward(self, dx):
        w, x, b, relu = self.cache

        dx = relu.backward(dx)

        self.weights[0][...] = x.T @ dx
        self.weights[1][...] = np.sum(dx, axis=0, keepdims=True)

        return dx @ w.T


class SkipConnection:
    def __init__(self):
        self.weights, self.grads = [], []

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout.copy()


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(x, 0)

    def backward(self, dout):
        x = self.cache
        return dout * (x > 0)


class Transformer:

    def __init__(self, vocab_size, wordvec_size=100, hidden_size=100):
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.weights = self.encoder.weights + self.decoder.weights
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs):
        result = self.encoder.forward(xs)
        # result2 = self.decoder.forward(result, )

    def backward(self, dout):
        pass


class TransformerDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        v, w, h = vocab_size, wordvec_size, hidden_size
        embed_W = (np.random.randn(v, w) / 100).astype(np.float32)
        lstm_Wx1 = (np.random.randn(w, 4 * h) / np.sqrt(w)).astype(np.float32)
        lstm_Wh1 = (np.random.randn(h, 4 * h) / np.sqrt(h)).astype(np.float32)
        lstm_b1 = np.zeros(4 * h).astype(np.float32)

        lstm_Wx2 = (np.random.randn(w, 4 * h) / np.sqrt(w)).astype(np.float32)
        lstm_Wh2 = (np.random.randn(h, 4 * h) / np.sqrt(h)).astype(np.float32)
        lstm_b2 = np.zeros(4 * h).astype(np.float32)

        affine_W = (np.random.randn(h, v) / np.sqrt(h)).astype(np.float32)
        affine_b = np.zeros(v).astype(np.float32)

        self.embed = TimeEmbedding(embed_W)
        self.lstm1 = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
        self.attention = TimeAttention()
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)


class MultiHeadAttention:
    def __init__(self, wordvec_size, hidden_size, num_heads):
        self.num_heads = num_heads
        self.d_k = wordvec_size // num_heads
        assert isinstance(self.d_k, int), f"{self.d_k} is not int"
        self.weights = [np.random.randn(wordvec_size, hidden_size),
                        np.random.randn(wordvec_size, hidden_size),
                        np.random.randn(wordvec_size, hidden_size),
                        np.random.randn(wordvec_size, hidden_size), ]
        self.cache = None

    def forward(self, query, key, value):
        batch_size, sequence_length, hidden_size = query.shape
        num_heads = self.num_heads
        d_k = self.d_k
        wq, wk, wv, wo = self.weights
        _query = query @ wq
        _key = key @ wk
        _value = value @ wv

        _queryHead = _query.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)
        _keyHead = _key.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)
        _valueHead = _value.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)

        attention_scores = (_queryHead @ _keyHead.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
        attention_output = attention_weights @ _valueHead

        attention_output_concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        output = attention_output_concat @ wo

        self.cache = (wq, wk, wv, wo, output, query, key, value)

        return output

    def backward(self):
        pass

    def split_heads(self, x):
        batch_size, sequence_length, _ = x.shape
        return x.reshape(batch_size, sequence_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, sequence_length, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)


def generate_addition(filename):
    seq = []
    with open(filename, "w") as fp:
        for i in range(250):
            for k in range(39):
                result = i + k
                string1 = f"{i}+{k}"
                string2 = f"_{result}"
                o1 = f"{string1:7}{string2:11}\n"
                seq.append(o1)
        random.shuffle(seq)
        fp.writelines(seq)


def generate_subtraction(filename):
    seq = []
    with open(filename, "w") as fp:
        for i in range(700, 950):
            for k in range(88, 127):
                result = i - k
                string1 = f"{i}-{k}"
                string2 = f"_{result}"
                o1 = f"{string1:7}{string2:11}\n"
                seq.append(o1)
        random.shuffle(seq)
        fp.writelines(seq)


def generate_division(filename):
    seq = []
    with open(filename, "w") as fp:
        for i in range(100, 7500):
            for k in range(1, 127):
                result = i / k
                string1 = f"{i}/{k}"
                string2 = f"_{round(result, 8)}"
                o1 = f"{string1:8}{string2:14}\n"
                seq.append(o1)
        random.shuffle(seq)
        fp.writelines(seq)


def generate_multiplication(filename):
    seq = []
    with open(filename, "w") as fp:
        for i in range(500, 750):
            for k in range(88, 127):
                result = i * k
                string1 = f"{i}*{k}"
                string2 = f"_{result}"
                o1 = f"{string1:7}{string2:13}\n"
                seq.append(o1)
        random.shuffle(seq)
        fp.writelines(seq)


def get_question_and_answer(*filename, train_ratio=0.9, reverse=True, gpu=True, torch=False, shuffle=True):
    o = ""
    for name in filename:
        with open(name, "r") as fp:
            string = fp.readlines()
        for k in string:
            o += k

    word_id, id_word, corpus = Preprocess.get_word_id(o)
    question = []
    answer = []
    question_and_answer = o.split('\n')
    if shuffle:
        random.shuffle(question_and_answer)
    o1 = []
    o2 = []
    for i in question_and_answer:
        if i != '':
            p = i.split('_')
            o1.append(p[0])
            o2.append(p[1])
    o3 = []
    for i in o2:
        i = '_' + i
        o3.append(i)
    if reverse:
        for i in o1:
            question.append(list(reversed(i)))
        for i in o3:
            answer.append(list(i))
    else:
        for i in o1:
            question.append(list(i))
        for i in o3:
            answer.append(list(i))
    for ls in question:
        for index, k in enumerate(ls):
            ls[index] = word_id[k]
    for ls in answer:
        for index, k in enumerate(ls):
            ls[index] = word_id[k]
    if not torch:
        train_index = int(len(question) * train_ratio)
        train_q = np.array(question[:train_index])
        test_q = np.array(question[train_index:])
        train_a = np.array(answer[:train_index])
        test_a = np.array(answer[train_index:])
        return train_q, test_q, train_a, test_a, word_id, id_word
    elif torch and not gpu:
        train_index = int(len(question) * train_ratio)
        train_q = tensor(question[:train_index]).long()
        test_q = tensor(question[train_index:]).long()
        train_a = tensor(answer[:train_index]).long()
        test_a = tensor(answer[train_index:]).long()
        return train_q, test_q, train_a, test_a, word_id, id_word
    elif torch and gpu:
        train_index = int(len(question) * train_ratio)
        train_q = tensor(question[:train_index]).long().to(device)
        test_q = tensor(question[train_index:]).long().to(device)
        train_a = tensor(answer[:train_index]).long().to(device)
        test_a = tensor(answer[train_index:]).long().to(device)
        return train_q, test_q, train_a, test_a, word_id, id_word


def evaluate(model, question, answer, word_id, id_word, size):
    answers = model.generate(question, word_id, size)
    print(''.join(reversed([id_word[int(i)] for i in question[0]])), '=', ''.join(id_word[int(i)] for i in answers))
    return 1 if answers == list(answer[:, 1:][0]) else 0


class Train:
    def __init__(self, model, optimizer):
        self.Tensorboard_logdir = None
        self.writer = None
        self.url = None
        self.browser = None
        self.tensorboard_process = None
        self._model, self._optimizer = model, optimizer

    def train(self, train_questions, train_answer, test_questions, test_answer, batch_size, max_epoch,
              word_id, id_word):
        max_iter = len(train_questions) // batch_size
        loss_cumulate = 0
        begin = time.time()
        size = len(train_answer[0, :]) - 1

        for epoch in range(max_epoch):
            correct = 0
            for iters in range(max_iter):
                batch_question = train_questions[iters * batch_size:(iters + 1) * batch_size]
                batch_answer = train_answer[iters * batch_size:(iters + 1) * batch_size]
                loss = self._model.forward(batch_question, batch_answer)
                self._model.backward()
                self._optimizer.update(self._model.weights, self._model.grads)

                loss_cumulate += loss

                if (iters + 1) % 10 == 0:
                    average_loss = loss_cumulate / 10
                    loss_cumulate = 0
                    self.print_result(epoch + 1, max_epoch, iters + 1, max_iter, begin, average_loss,
                                      self._optimizer.lr)
            for i in range(len(test_questions)):
                question, answer = test_questions[[i]], test_answer[[i]]
                correct += evaluate(self._model, question, answer, word_id, id_word, size)
            if self.writer is not None:
                self.writer.add_scalar("Correctness", correct / len(test_questions), epoch)
        if self.writer is not None and self.tensorboard_process is not None:
            self.writer.close()
            self.tensorboard_process.terminate()

    def PYTORCH_train(self, train_question, train_answers, test_question, test_answers, batch_size,
                      max_epoch, word_id, id_word, log=True, log_dir=None, Tensorboard_reloadInterval=30,
                      log_file_name=''):
        max_iter = len(train_question) // batch_size
        loss_cumulate = 0
        begin = time.time()
        size = len(train_answers[0, :]) - 1
        average_loss = 0
        if log:
            self.open_tensorboard(log_dir, Tensorboard_reloadInterval, f"({log_file_name})")

        for epoch in range(max_epoch):
            correct = 0
            for iters in range(max_iter):
                batch_question = train_question[iters * batch_size:(iters + 1) * batch_size]
                batch_answer = train_answers[iters * batch_size:(iters + 1) * batch_size]
                self._optimizer.zero_grad()
                loss = self._model.forward(batch_question, batch_answer)
                loss.backward()
                self._optimizer.step()

                loss_cumulate += loss

                if iters % 10 == 0:
                    average_loss = loss_cumulate / 10
                    loss_cumulate = 0
                    self.print_result(epoch, max_epoch, iters, max_iter, begin, average_loss,
                                      self._optimizer.param_groups[0]['lr'])
            self.print_result(epoch, max_epoch, max_iter, max_iter, begin, average_loss,
                              self._optimizer.param_groups[0]['lr'])
            for i in range(len(test_question)):
                question, answer = test_question[[i]], test_answers[[i]]
                correct += evaluate(self._model, question, answer, word_id, id_word, size)
            if self.writer is not None:
                self.writer.add_scalar("Correctness", correct / len(test_question), epoch)

        self.print_result(max_epoch, max_epoch, max_iter, max_iter, begin, average_loss,
                          self._optimizer.param_groups[0]['lr'])
        if self.writer is not None and self.tensorboard_process is not None:
            self.writer.close()
            self.tensorboard_process.terminate()

    def open_tensorboard(self, log_dir=None, Tensorboard_reloadInterval=30, log_file_name=''):
        directory = None
        if log_dir is not None:
            from datetime import datetime
            import socket
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            directory = os.path.join(
                os.getcwd(), f"{log_dir}", current_time + "_" + socket.gethostname()
            )
            log_dir = os.path.join(os.getcwd(), f"{log_dir}")
        self.writer = writer = SummaryWriter(log_dir=directory, filename_suffix=log_file_name)
        self.Tensorboard_logdir = writer.log_dir
        writer.add_scalar("Correctness", 0, 0)
        writer.flush()
        self.open_Tensorboard(log_dir, Tensorboard_reloadInterval)

    @staticmethod
    def open_Tensorboard(log_dir=None, Tensorboard_reloadInterval=30):
        log_dir = rf'{os.getcwd()}\runs' if log_dir is None else log_dir
        tensorboard_port = 6006
        print('copy to run:', ''.join(['tensorboard', f' --logdir={log_dir}', f' --port={tensorboard_port}',
                                       f' --reload_interval={Tensorboard_reloadInterval}']))

    @staticmethod
    def print_result(current_epoch, total_epoch, current_iters, total_iters, begin, loss, lr, round=None):
        blocks = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
        list1 = list('│' + ' ' * 20 + '│')  # length 22 last_index: 21
        list2 = list('│' + ' ' * 20 + '│')  # length 22 last_index: 21

        # epoch index should be 1 <= index <= 20 (200 updates)
        epoch_percentage = current_epoch * 100 / total_epoch
        iter_percentage = current_iters * 100 / total_iters

        epoch_index = min(int(epoch_percentage / 5), 20)
        iter_index = min(int(iter_percentage / 5), 20)
        epoch_fine = (int(epoch_percentage / 0.625) if epoch_percentage % 0.625 == 0 else math.floor(
            epoch_percentage / 0.625)) % 8
        iter_fine = (int(iter_percentage / 0.625) if iter_percentage % 0.625 == 0 else math.floor(
            iter_percentage / 0.625)) % 8

        list1[1:1 + epoch_index] = '█' * len(list1[1:1 + epoch_index])
        list2[1:1 + iter_index] = '█' * len(list2[1:1 + iter_index])
        if epoch_index < 20: list1[1 + epoch_index] = blocks[epoch_fine]
        if iter_index < 20: list2[1 + iter_index] = blocks[iter_fine]
        string1 = ''.join(list1)
        string2 = ''.join(list2)

        time1 = time.time() - begin
        print(f'\rEpoch: {string1} {format(epoch_percentage, ".2f")}% |'
              f' Iters: {string2} {format(iter_percentage, ".2f")}% | Time: {time1:.3f}s | loss: {loss:.3f} | lr: {lr:.3f}'
              f' | current round: {round}',
              end='',
              flush=True)


class printProcess:
    def __init__(self):
        self.ls = []
        self.metrics = []
        self.percentage = None

    def print_result(self, *args, begin=None, timing=True):
        """ format: (current, total), metrics"""
        ls = self.ls
        blocks = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
        percentage = []
        # epoch index should be 1 <= index <= 20 (200 updates)
        count = 0
        for i in args:
            if isinstance(i, tuple):
                percentage.append(i[0] * 100 / i[1])
                count += 1
        assert count == len(ls), "bar not match the arguments input"
        index = [min(int(percent / 5), 20) for percent in percentage]

        fine = [(int(percent / 0.625) if percent % 0.625 == 0 else math.floor(percent / 0.625)) % 8 for percent in
                percentage]

        for idx, (title, bar) in enumerate(ls):
            bar[1:1 + index[idx]] = '█' * len(bar[1:1 + index[idx]])
            if index[idx] < 20:
                bar[1 + index[idx]] = blocks[fine[idx]]

        string = "\r"
        for idx, (title, bar) in enumerate(ls):
            string += f" ▏{title}: {''.join(bar)} {format(percentage[idx], '.2f')}%"

        string2 = ""

        if begin is not None and timing:
            string2 += f" ▏Time: {format(time.time() - begin, '.2f')} s"
        for idx, (title, unit) in enumerate(self.metrics):
            string2 += f" ▏{title}: {args[count + idx]} {unit if unit is not None else ''}"

        print(string + string2, end='', flush=True)

    def add_bar(self, *title):
        for i in title:
            self.ls.append((i, list('│' + ' ' * 20 + '│')))

    def add_metrics(self, *metrics):
        for i in metrics:
            if isinstance(i, str):
                i = (i, None)
            if len(i) == 1:
                self.metrics.append((i[0], None))
            else:
                self.metrics.append(i)


if __name__ == "__main__":
    pass
    # from sklearn.model_selection import RandomizedSearchCV
    #
    # # 定义超参数搜索范围
    # round0 = 0
    # train_questions, test_questions, train_answer, test_answer, _word_id, _id_word = get_question_and_answer(
    #     "division_shuffle.txt")
    # # 初始化你的模型
    # vocab_size = len(_word_id)
    # wordvec_size = 16
    # hidden_size = 128
    # _max_epoch = 200
    # _max_grad = 5.0
    # model = AttentionSeq2seq(_word_id, vocab_size, wordvec_size, hidden_size)
    # # 创建随机搜索对象
    # param_dist = {
    #     'word_id': [_word_id],
    #     'vocab_size': [vocab_size],
    #     'wordvec_size': [random.randint(8, 8) for i in range(1)],
    #     'hidden_size': [random.randint(8, 8) for k in range(1)]
    # }
    # random_search = RandomizedSearchCV(
    #     model, param_distributions=param_dist, n_iter=50, cv=5)
    # fit_params = {
    #     'X_test': test_questions,
    #     'y_test': test_answer,
    #     'optimizer': Adam(),
    #     'batch_size': 2048,
    #     'max_grad': _max_grad,
    #     'max_epoch': _max_epoch
    # }
    # #
    # random_search.fit(train_questions, train_answer, **fit_params)
    #
    # best_params = random_search.best_params_
    # print("Best parameters found: ", best_params)
    # results = random_search.cv_results_
    # plt.plot(results['mean_train_score'], label='train')
    # plt.plot(results['mean_test_score'], label='validation')
    # plt.xlabel('Hyperparameter combinations')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig('result.png')
