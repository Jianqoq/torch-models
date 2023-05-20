import inspect

import numpy

import server
import numpy as np
import graphviz
import random
from time import time

import matplotlib.pyplot as plt


class Function:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.vars = 1
        self.has_conv = 0
        self.repeats = None
        self.val = self.result()
        self.grad_name = 0
        self.holder = None
        self.T = self.val.T
        self.graph = None
        self.grad = None
        self.parent = None
        self.son = None
        self.index = None
        self.label = None
        self.x1 = None
        self.x2 = None
        self.abbreviation = None
        self.size = self.val.size
        self.shape = self.val.shape

    def update(self):
        self.val = self.result()

    def update_label(self):
        pass

    @staticmethod
    def plot(*func, start=-10, end=10):
        pass

    def result(self) -> np.ndarray:
        """
        require to implement by yourself
        """
        pass

    def get_grad(self, grad):
        """
        require to implement by yourself
        """
        pass

    def get_graph(self):
        return

    def get_label(self):
        pass

    def visualize(self, open_file=False, file_name=None, size=None, web=True):
        total = search_nodes(self)
        p = View(self, head=True, time_begin=time(), total_task=total, filename=file_name)
        if size is not None:
            p.graph_object.graph_attr['size'] = size
        if open_file:
            if web:
                assert "json" in file_name, "web have to use json file"
                p.graph_object.render(filename=p.graph_object.filename, format=p.graph_object.format)
                server.start_server(file_name)
            else:
                p.graph_object.view()
        else:
            return p.graph_object

    def gradient(self,
                 grad: np.ndarray | None = None,
                 create_graph=False,
                 ):
        stack = inspect.stack()
        grad_name = stack[1].code_context[0].split(",")[0].split('(')[1]
        if create_graph:
            self.grad_name = grad_name
        if grad is None:
            grad = np.array(1.).reshape(self.val.shape)
        Prime(self, grad=grad, create_graph=create_graph, grad_name=self.grad_name)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *axis):
        return transpose(self, axis if len(axis) > 0 else None)

    def get_values(self, x, y):
        """
        :param x: Any
        :param y: Any
        :return: (x, y)
        """
        try:
            if x.vars == 3:
                self.has_conv += 1
            elif x.has_conv:
                self.has_conv += x.has_conv
            x = x.val
        except:
            pass
        try:
            if y.vars == 3:
                self.has_conv += 1
            elif y.has_conv:
                self.has_conv += y.has_conv
            y = y.val
        except:
            pass
        return x, y

    def get_value(self, x):
        """
        :param x: Any
        :return: x
        """
        if hasattr(x, "vars"):
            if x.vars == 3:
                self.has_conv += 1
            elif x.has_conv:
                self.has_conv += x.has_conv
            return x.val
        else:
            return x

    def sum(self):
        pass

    def __str__(self):
        expression = np.array2string(
            self.val, prefix=f"{self.__class__.__name__}(", separator=', ')
        return f"{self.__class__.__name__}({expression}, shape={self.shape})"

    def __neg__(self):
        return Multi(Matrix(-1), self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)

    def __matmul__(self, other):
        return Matmul(self, other)

    def __rmatmul__(self, other):
        return Matmul(other, self)

    def __iter__(self):
        for index, val in enumerate(self.val):
            yield Matrix(val, f"Mat{index}")

    def __abs__(self):
        return Abs(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        x = self.val[item]
        return Slice(x, self, item)

    def __min__(self):
        return Min(self)

    def __len__(self):
        return len(self.val)


class tensordot(Function):

    def __init__(self, x, y, axes=2):
        self.axes = axes
        self.dot_shape = None
        self.yt = None
        self.xt = None
        self.newshape_a = None
        self.newshape_b = None
        self.oldaxes_a = None
        self.oldaxes_b = None
        self.xsaved_shape = None
        self.ysaved_shape = None
        self.newaxes_a = None
        self.newaxes_b = None
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        grad = grad.reshape(self.dot_shape)
        at_grad = grad @ self.yt.T
        bt_grad = self.xt.T @ grad
        at_grad = at_grad.reshape(self.newshape_a)
        bt_grad = bt_grad.reshape(self.newshape_b)
        return at_grad.transpose(self.oldaxes_a), bt_grad.transpose(self.oldaxes_b)

    def result(self):
        """ code from numpy source code"""
        a, b = self.get_values(self.x, self.y)
        axes = self.axes
        try:
            iter(axes)
        except Exception:
            axes_a = list(range(-axes, 0))
            axes_b = list(range(0, axes))
        else:
            axes_a, axes_b = axes
        try:
            na = len(axes_a)
            axes_a = list(axes_a)
        except TypeError:
            axes_a = [axes_a]
            na = 1
        try:
            nb = len(axes_b)
            axes_b = list(axes_b)
        except TypeError:
            axes_b = [axes_b]
            nb = 1

        as_ = a.shape
        nda = a.ndim
        bs = b.shape
        ndb = b.ndim
        equal = True
        if na != nb:
            equal = False
        else:
            for k in range(na):
                if as_[axes_a[k]] != bs[axes_b[k]]:
                    equal = False
                    break
                if axes_a[k] < 0:
                    axes_a[k] += nda
                if axes_b[k] < 0:
                    axes_b[k] += ndb
        if not equal:
            raise ValueError("shape-mismatch for sum")

        # Move the axes to sum over to the end of "a"
        # and to the front of "b"
        notin = [k for k in range(nda) if k not in axes_a]
        newaxes_a = notin + axes_a
        N2 = 1
        for axis in axes_a:
            N2 *= as_[axis]
        self.xsaved_shape = newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
        olda = [as_[axis] for axis in notin]

        notin = [k for k in range(ndb) if k not in axes_b]
        newaxes_b = axes_b + notin
        N2 = 1
        for axis in axes_b:
            N2 *= bs[axis]
        self.ysaved_shape = newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
        oldb = [bs[axis] for axis in notin]

        self.oldaxes_a = newaxes_a
        self.oldaxes_b = newaxes_b

        cache1 = a.transpose(newaxes_a)
        cache2 = b.transpose(newaxes_b)

        self.newshape_a = cache1.shape
        self.newshape_b = cache2.shape

        a = cache1.reshape(newshape_a)
        b = cache2.reshape(newshape_b)
        self.xt = a
        self.yt = b
        out = a @ b
        self.dot_shape = out.shape
        return out.reshape(olda + oldb)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        grad = self.holder.reshape(self.dot_shape)
        at_grad = grad @ (self.y.transpose(*self.oldaxes_b).reshape(self.ysaved_shape)).transpose()
        bt_grad = (self.x.transpose(*self.oldaxes_a).reshape(self.xsaved_shape)).transpose() @ grad
        return at_grad.reshape(self.newshape_a).transpose(*self.oldaxes_a), bt_grad.reshape(self.newshape_b).transpose(*self.oldaxes_b)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.label = f"np.tensordot({labels[0]}, {labels[1]}, axes={self.axes})"


class Matrix(Function):

    def __init__(self, data, label=None):
        x = data if isinstance(data, np.ndarray) else np.array(data)
        super().__init__(x)
        self.vars = 0
        self.label = label
        self.shape = x.shape
        self.T = x.T
        self.size = x.size

    def result(self):
        return self.x

    def update(self):
        self.val = self.x

    def set_x(self, x):
        self.x = x

    def auto_label(self, data):
        frame = inspect.currentframe()
        try:
            outer_frame = frame.f_back
            for i in outer_frame.f_locals.items():
                if id(i[1]) == id(data):
                    self.label = i[0]
        finally:
            del frame

    def get_label(self):
        return self.label

    def repeat(self, times, axis):
        return repeat(self, times, axis)

    def __len__(self):
        return len(self.x)


class Slice(Function):

    def __init__(self, data, origin_data, index):
        super().__init__(data)
        self.x = origin_data
        self.sliced_data = data
        self._shape = data.shape
        self._index = index

    def get_grad(self, grad=None):
        """
        :param grad: same shape as the sliced array shape
        :return: ndarray
        """
        assert grad.shape == self._shape, f"grad shape {grad.shape} doesn't match {self._shape}"
        zeros = np.zeros(self.x.shape)
        zeros[self._index] = grad
        return zeros

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 grad_name="",
                 create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        elif isinstance(grad, np.ndarray):
            pass
        else:
            grad = np.array(grad)
        Prime(self, grad=grad)

    def result(self):
        return self.get_value(self.x)


class stack(Function):
    def __init__(self, data, axis):
        self.axis = axis
        super().__init__(data)
        self.vars = 5
        self.size = self.val.size

    def get_grad(self, grad=None):

        pass

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        graph = (self.holder@self.y.T, self.x@self.holder)
        return graph

    def get_label(self):
        self.x1 = "Iter"
        self.label = f"np.stack({self.x1}, axis={self.axis})"

    def result(self):
        val = []
        for index, i in enumerate(self.x):
            i.index = index
            val.append(self.get_value(i))
        return np.stack(val, axis=self.axis)


class EqualSlice(Function):

    def __init__(self, origin_data, num, axis):
        self.num = num
        self.axis = axis
        super().__init__(origin_data)
        self.vars = 4
        self._shape = origin_data.shape
        self.x3 = self.get_real()
        self.grad = None
        self.cumulate = 0

    def get_grad(self, grad=None):
        """
        :param grad: same shape as the sliced array shape
        :return: ndarray
        """
        if grad is None:
            grad = [i.grad if i.grad is not None else np.zeros(i.shape) for i in self.x3]
            grad = np.stack(grad, axis=self.axis).reshape(self.x.shape)
            for i in self.x3:
                i.grad = None
            return check_shape(grad, self.x)
        else:
            grad = [i for i in grad]
            grad = np.stack(grad, axis=self.axis).reshape(self._shape)
            return check_shape(grad, self.x)

    def gradient(self,
                 grad: list | tuple | None = None,
                 grad_name="",
                 create_graph=False):
        if grad is None:
            pass
        else:
            Prime(self, grad=grad)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return stack(self.holder, axis=self.axis).reshape(self._shape),

    def get_label(self):
        self.x1 = [i.label for i in self.x3]
        self.label = f"EqualSlice({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.array([np.squeeze(i) for i in np.split(val, self.num, axis=self.axis)])

    def get_real(self):
        ls = []
        for index, val in enumerate(self.val):
            a = Matrix(val)
            a.vars = 3
            a.parent = self
            a.label = f"converge{index}"
            ls.append(a)
        return ls

    def __iter__(self):
        return iter(self.x3)


class reshape(Function):

    def __init__(self, data, shape: list | tuple):
        self.shape = shape
        super().__init__(data)
        self.x = data
        self._shape = data.shape
        self.size = data.size

    def get_grad(self, grad):
        assert grad.size == self.size
        return grad.reshape(self._shape)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return reshape(self.holder, shape=self._shape),

    def get_label(self):
        self.x1 = get_label(self.x)[0]
        self.label = f"np.reshape({self.x1}, {self.shape})"

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 grad_name="",
                 create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad)

    def result(self):
        return np.reshape(self.x.val, self.shape)


class Matmul(Function):

    def __init__(self, x, y):
        self._brocased = False
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        if self._brocased:
            return grad @ self.y.val.transpose(0, 1, 3, 2), self.x.val.transpose(0, 1, 3, 2) @ grad
        return check_shape(grad @ self.y.T, self.x), check_shape(self.x.T @ grad, self.y)

    def result(self):
        val1, val2 = self.get_values(self.x, self.y)
        if val1.ndim > 2:
            self._brocased = True
        return np.matmul(val1, val2)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        if self._brocased:
            return self.holder @ self.y.transpose(0, 1, 3, 2), self.x.transpose(0, 1, 3, 2) @ self.holder
        graph = (self.holder@transpose(self.y), transpose(self.x)@self.holder)
        return graph

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.label = f"({labels[0]} @ {labels[1]})"


class starmulti(Function):

    def __init__(self, x: Matrix, y: Matrix):
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        val1, val2 = self.get_values(self.x, self.y)
        grad1 = val2 * grad
        grad2 = grad * val1
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.x1 = labels[0]
        self.x2 = labels[1]
        if isinstance(self.x, (Add, sub, Divide)) and not isinstance(self.y, (Add, sub, Divide)):
            self.label = f"({self.x1}) * {self.x2}"
        elif not isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
            self.label = f"{self.x1} * ({self.x2})"
        elif isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
            self.label = f"({self.x1}) * ({self.x2})"
        else:
            self.label = f"{self.x1} * {self.x2}"

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        graph = (starmulti(self.y, self.holder), starmulti(self.holder, self.x))
        return graph

    def result(self):
        val1, val2 = self.get_values(self.x, self.y)
        return val1 * val2


class transpose(Function):

    def __init__(self, x, axis = None):
        self.axis = axis
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad.T if self.axis is None else grad.transpose(self.axis)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        result = Matrix(self.holder.T) if self.axis is None else self.holder.transpose(*self.axis)
        return result,

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"{self.x1}.T" if self.axis is None else f"np.transpose({self.x1}, axes={self.axis})"

    def result(self):
        val = self.get_value(self.x)
        return val.T if self.axis is None else val.transpose(self.axis)


class trace(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.shape = None

    def get_grad(self, grad):
        val = self.get_value(self.x)
        assert val.shape[-2] == val.shape[-1], "input has to be square matrix"
        grad = grad * np.identity(val.shape[0])
        self.shape = val.shape[0]
        return check_shape(grad, val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder*Matrix(np.identity(self.shape), f"eye({self.shape})"),

    def result(self):
        val = self.get_value(self.x)
        return np.trace(val)


class inv(Function):

    def __init__(self, x):
        super().__init__(x)
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"inv({self.x1})"

    def get_grad(self, grad):
        val = self.get_value(self.x)
        assert val.shape == grad.shape, f"shape {val.shape} != grad shape {grad.shape}"
        temp = np.linalg.inv(val)
        grad = np.transpose(temp @ np.transpose(grad) @ (-temp))  # transpose??
        return check_shape(grad, val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return transpose(inv(self.x)@transpose(self.holder)@(Matrix(-1, "(-1)")*inv(self.x))),

    def result(self):
        val = self.get_value(self.x)
        op = np.linalg.inv
        return op(val)


class Add(Function):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        grad1, grad2 = grad.copy(), grad.copy()
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.x1 = labels[0]
        self.x2 = labels[1]
        self.label = f"{self.x1} + {self.x2}"

    def result(self):
        result1, result2 = self.get_values(self.x, self.y)
        return np.add(result1, result2)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder, self.holder


class sum(Function):

    def __init__(self, x, axis=None, keepdims=None):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def get_grad(self, grad):
        shape = [*self.x.val.shape]
        shape[self.axis] = 1
        grad = grad.reshape(shape)
        grad = grad.repeat(self.x.val.shape[self.axis], axis=self.axis)
        return check_shape(grad, self.x)

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        axis = f", axis={self.axis}" if self.axis is not None else ""
        keepdims = f", keepdims={self.keepdims}" if self.keepdims is not None else ""
        self.label = f"np.sum({self.x1}{axis}{keepdims})"

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        shape = [*self.x.val.shape]
        shape[self.axis] = 1
        return graph_check_shape(repeat(reshape(self.holder, shape), self.x.val.shape[self.axis], axis=self.axis), self.x),

    def result(self):
        return np.sum(self.get_value(self.x), axis=self.axis, keepdims=self.keepdims)


class ScalarToMatrix(Function):
    def __init__(self, x, shape):
        super().__init__(x)
        self.shape = shape

    def get_grad(self, grad):
        return np.sum(grad)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return sum(self.holder)

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"ScalarToMatrix({self.x1})"

    def result(self):
        result = self.get_value(self.x)
        return np.broadcast_to(result, self.shape)


class sub(Function):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        grad1, grad2 = -1 * grad, grad
        return check_shape(grad1, self.x), check_shape(
            grad2, self.y)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.x1 = labels[0]
        self.x2 = labels[1]
        self.label = f"({self.x1} - ({self.x2}))" if self.x2.startswith("-") else f"({self.x1} - {self.x2})"

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder, starmulti(self.holder, Matrix([-1.], "(-1)"))

    def result(self):
        result1, result2 = self.get_values(self.x, self.y)
        return np.subtract(result1, result2)


class repeat(Function):

    def __init__(self, x, times: int, axis: int):
        self.axis = axis
        self.times = times
        super().__init__(x)

    def get_grad(self, grad):
        grad = np.sum(grad, axis=self.axis, keepdims=True)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return sum(self.holder, axis=self.axis, keepdims=True),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.repeat({self.x1}, repeats={self.times}, axis={self.axis})"

    def result(self):
        val = self.get_value(self.x)
        return np.repeat(val, self.times, axis=self.axis)


class Max(Function):

    def __init__(self, x, axis: int | None = None):
        self.axis = axis
        super().__init__(x)
        self.x = x
        self.label = None

    def get_grad(self, grad):
        assert self.x.shape == self.val.shape, f"self.x shape{self.x.shape} != self.val shape {self.val.shape}"
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask / div
        return check_shape(grad * new, self.x)

    def result(self):
        result = self.get_value(self.x)
        return np.amax(
            result,
            axis=self.axis) if self.axis is not None else np.max(result)


class Min(Function):

    def __init__(self, x, axis: int | None = None):
        self.axis = axis
        super().__init__(x)
        self.x = x
        self.label = None

    def get_grad(self, grad):
        assert self.x.shape == self.val.shape, f"self.x shape{self.x.shape} != self.val shape {self.val.shape}"
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask / div
        return check_shape(grad * new, self.x)

    def result(self):
        result = self.get_value(self.x)
        return np.amax(
            result,
            axis=self.axis) if self.axis is not None else np.max(result)


class Multi(Function):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        grad1, grad2 = grad * self.y.val, grad * self.x.val
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return Multi(self.holder, self.y), Multi(self.holder, self.x)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.x1 = labels[0]
        self.x2 = labels[1]
        if self.x1 == self.x2:
            self.label = f"{self.x1}**2"
        elif isinstance(self.x, (Add, sub, Divide)) and not isinstance(self.y, (Add, sub, Divide)):
            self.label = f"({self.x1})*{self.x2}"
        elif not isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
            self.label = f"{self.x1}*({self.x2})"
        elif isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
            self.label = f"({self.x1})*({self.x2})"
        elif self.x1 == "1":
            self.label = f"{self.x2}"
        elif self.x2 == "1":
            self.label = f"{self.x1}"
        else:
            self.label = f"{self.x1}*{self.x2}"

    def result(self):
        result1, result2 = self.get_values(self.x, self.y)
        return np.multiply(result1, result2)


class Sigmoid(Function):
    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        return grad * self.val * (1. - self.val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * self * (Matrix(1., "1") - self),

    def get_label(self):
        if self.label is None:
            labels = get_label(self.x)
            self.x1 = labels[0]
            self.label = f"sigmoid({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return 1. / (1. + np.exp(-val))

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y)
        plt.show()


class SoftmaxWithLoss(Function):
    def __init__(self, x, dim):
        self.dim = dim - 1
        super().__init__(x)

    def get_grad(self, grad):
        return grad * self.val * (1 - self.val)

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"Softmax({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        x = val - val.max(axis=self.dim, keepdims=True)
        x_exp = np.exp(x)
        result = np.sum(x_exp, axis=self.dim, keepdims=True)
        x = x_exp / result
        return x

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x) / np.sum(np.exp(x))
        plt.plot(x.get(), y.get())
        plt.show()


class softmax(Function):
    def __init__(self, x, dim):
        self.dim = dim - 1
        super().__init__(x)

    def get_grad(self, grad):
        dx = self.val * grad
        sumdx = np.sum(dx, axis=self.dim, keepdims=True)
        dx -= self.val * sumdx
        return check_shape(dx, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        dx = self * self.holder
        return dx - self * sum(dx, axis=self.dim, keepdims=True),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"softmax({self.x1}, dims={self.dim})"

    def result(self):
        val = self.get_value(self.x)
        x = val - val.max(axis=self.dim, keepdims=True)
        x_exp = np.exp(x)
        result = np.sum(x_exp, axis=self.dim, keepdims=True)
        x = x_exp / result
        return x

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x) / np.sum(np.exp(x))
        plt.plot(x.get(), y.get())
        plt.show()


class mean(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = grad / np.size(self.x.x)
        grad = np.full(self.x.x.shape, grad)
        return grad

    def result(self):
        val = self.get_value(self.x)
        return np.mean(val)


class Divide(Function):

    def __init__(self, x, y):
        self._brocasted = False
        super().__init__(x, y)
        self.vars = 2

    def get_grad(self, grad):
        grad1 = grad / (self.y.val + 1e-8)
        grad2 = grad * (-self.x.val / (np.square(self.y.val) + 1e-8))
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return graph_check_shape(self.holder/self.y, self.x), graph_check_shape(self.holder * ((Matrix(-1., "(-1)")*self.x) / square(self.y)), self.y)

    def get_label(self):
        labels = get_label(self.x, self.y)
        self.x1 = labels[0]
        self.x2 = labels[1]
        if isinstance(self.x, (Add, sub)) and not isinstance(self.y, (Add, sub)):
            self.label = f"(({self.x1})/{self.x2})"
        elif not isinstance(self.x, (Add, sub)) and isinstance(self.y, (Add, sub)):
            self.label = f"({self.x1}/({self.x2}))"
        elif isinstance(self.x, (Add, sub)) and isinstance(self.y, (Add, sub)):
            self.label = f"(({self.x1})/({self.x2}))"
        else:
            self.label = f"({self.x1}/{self.x2})"

    def result(self):
        result1, result2 = self.get_values(self.x, self.y)
        return result1 / result2


class inner(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        labels = get_label(x, y)
        self.x1 = labels[0]
        self.x2 = labels[1]

    def get_grad(self, grad):
        grad1 = np.array(grad) / (self.y.val + 1e-8)
        grad2 = np.array(grad) * (-self.x.val / (np.square(self.y.val) + 1e-8))
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder/self.y, self.holder * Matrix(-1., "(-1)")*self.x / square(self.y)

    def result(self):
        result1, result2 = self.get_values(self.x, self.y)
        return np.inner(result1, result2)


class exp(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * np.exp(self.x.val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        temp = exp(self.x)
        temp.label = self.label
        return self.holder * temp,

    def get_label(self):
        labels = get_label(self.x)
        self.label = f"np.exp({labels[0]})"

    def result(self) -> np.ndarray:
        val = self.get_value(self.x)
        return np.exp(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x)
        plt.plot(x, y)
        plt.show()


class pow(Function):

    def __init__(self, x, power):
        self.power = power
        super().__init__(x, power)

    def get_grad(self, grad):
        grad1 = grad * self.power * np.power(self.x.val, self.power - 1)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * self.power * pow(self.x, self.power - 1),

    def get_label(self):
        self.label = f"{get_label(self.x)[0]}**{self.power}"

    def result(self):
        val = self.get_value(self.x)
        return np.power(val, self.power)

    @staticmethod
    def plot(self=None, power=2, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.power(x, power)
        plt.plot(x, y)
        plt.show()


class sin(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad1 = grad * np.cos(self.x.val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * cos(self.x),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.sin({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.sin(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.sin(x)
        plt.plot(x, y)
        plt.show()


class sec(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * np.tan(self.x.val) * (1. / np.cos(self.x.val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        temp = sec(self.x)
        temp.label = self.label
        return self.holder * tan(self.x) * temp,

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"sec({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return 1. / np.cos(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = 1. / np.cos(x)
        plt.plot(x, y)
        plt.show()


class sinh(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * np.cosh(self.x.val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * cosh(self.x),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"sinh({labels[0]})"

    def result(self):
        val = self.get_value(self.x)
        return np.sinh(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.sinh(x)
        plt.plot(x, y)
        plt.show()


class arcsin(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * (1. / np.sqrt(1. - np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * (Matrix(-1., "-1") / sqrt(Matrix(1., "1") - square(self.x))),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.arcsin({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.arcsin(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.arcsin(x)
        plt.plot(x, y)
        plt.show()


class arcsec(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * (1. / (np.abs(self.x.val)*np.sqrt(np.square(self.x.val) - 1.)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder / Abs(self.x)*sqrt(square(self.x) + Matrix(-1, "(-1)")),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.arcsec({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.arccos(1./val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.arccos(x)
        plt.plot(x, y)
        plt.show()


class ln(Function):

    def __init__(self, x):
        super().__init__(x)

    def update_label(self):
        self.label = f"ln({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / self.x.val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder / self.x,

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.log({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        return np.log(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.log(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class log(Function):
    """
    any base log
    """

    def __init__(self, base, x):
        self.base = base
        super().__init__(x)
        assert base != 0, "base can't be zero"

    def get_grad(self, grad):
        grad = grad * (1 / (self.x.val * np.log(self.base)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder/(self.x * ln(self.base)),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"(np.log({labels[0]}) / np.log({self.x1}))"

    def result(self):
        val = self.get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        return np.log(val) / np.log(self.base)

    @staticmethod
    def plot(base=10, *func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.log(x) / np.log(base)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cos(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad1 = -grad * np.sin(self.x.val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return Matrix(-1, "(-1)")*self.holder * sin(self.x),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.cos({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.cos(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.cos(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cosh(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * np.sinh(self.x.val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * sinh(self.x),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.cosh({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.cosh(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.cosh(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arcos(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * (-1. / np.sqrt(self.x.val + 1.))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * (Matrix(-1, "(-1)") / sqrt(self.x + Matrix(1, "1"))),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.arccos({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.arccos(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arccos(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class csc(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = -grad * np.csc(self.x.val) * cot(self.x.val).val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        temp = csc(self.x)
        temp.label = self.label
        return self.holder * Matrix(-1, "(-1)") * temp * cot(self.x),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"(1 / np.sin({labels[0]}))"

    def result(self):
        val = self.get_value(self.x)
        return 1. / np.sin(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = 1 / np.sin(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class tan(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * (np.square(sec(self.x.val).val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * sec(self.x) ** 2,

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.tan({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.tan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.tan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arctan(Function):

    def __init__(self, x):
        super().__init__(x)

    def update_label(self):
        self.label = f"arctan({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / (1 + np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder / (Matrix(1, "1") + square(self.x)),

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.arctan({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.arctan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arctan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class tanh(Function):

    def __init__(self, x):
        super().__init__(x)

    def get_grad(self, grad):
        grad = grad * (1.0 - np.square(self.val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        temp = tanh(self.x)
        temp.label = self.label
        return self.holder * (Matrix(1, "1") - temp**2),

    def get_label(self):
        if self.label is None:
            labels = get_label(self.x)
            self.x1 = labels[0]
            self.label = f"np.tanh({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.tanh(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.tanh(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cot(Function):

    def __init__(self, x, label_on=False):
        super().__init__(x)
        if label_on:
            labels = get_label(x)
            self.label = f"cot({labels[0]})"

    def get_grad(self, grad):
        grad = grad * np.square(csc(self.x).val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * square(csc(self.x)),

    def get_label(self):
        if self.label is None:
            labels = get_label(self.x)
            self.x1 = labels[0]
            self.label = f"(1 / np.tan({self.x1}))"

    def result(self):
        val = self.get_value(self.x)
        return 1 / np.tan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = 1 / np.tan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arcot(Function):

    def __init__(self, x, label_on=False):
        super().__init__(x)
        if label_on:
            labels = get_label(x)
            self.label = f"arcot({labels[0]})"

    def get_grad(self, grad):
        grad = grad * (-1 / (1 + np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * (Matrix(-1, "(-1)") / (Matrix(1, "1") + square(self.x))),

    def get_label(self):
        if self.label is None:
            labels = get_label(self.x)
            self.x1 = labels[0]
            self.label = f"np.arctan({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.arctan(1 / val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arctan(1 / x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class square(Function):

    def __init__(self, x, label_on=False):
        super().__init__(x)
        if label_on:
            labels = get_label(x)
            self.x1 = labels[0]
            self.label = f"{self.x1}**2"

    def get_grad(self, grad):
        val = self.get_value(self.x)
        grad = grad * 2 * val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * Matrix(2., "2") * self.x,

    def get_label(self):
        if self.label is None:
            labels = get_label(self.x)
            self.x1 = labels[0]
            self.label = f"np.square({self.x1})"

    def result(self):
        val = self.get_value(self.x)
        return np.square(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.square(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class sqrt(Function):

    def __init__(self, x, label_on=False):
        super().__init__(x)
        if label_on:
            labels = get_label(x)
            self.x1 = labels[0]
            self.label = f"sqrt({self.x1})"

    def get_grad(self, grad):
        val = self.get_value(self.x)
        grad = grad * 0.5 * np.power(val, -0.5)
        return check_shape(grad, self.x)

    def get_label(self):
        labels = get_label(self.x)
        self.x1 = labels[0]
        self.label = f"np.sqrt({self.x1})"

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * Matrix(0.5, "0.5") * pow(self.x, -0.5),

    def result(self):
        val = self.get_value(self.x)
        return np.sqrt(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.sqrt(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class argmax(Function):

    def __init__(self, x, label_on=False):
        super(argmax, self).__init__(x)
        if label_on:
            self.label = x.label

    def result(self):
        val = self.get_value(self.x)
        return np.argmax(val)

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 grad_name="",
                 create_graph=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class argmin(Function):

    def __init__(self, x):
        super(argmin, self).__init__(x)
        self.x = x

    def result(self):
        val = self.get_value(self.x)
        return np.argmin(val)

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class Abs(Function):

    def __init__(self, x):
        super(Abs, self).__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"abs({labels[0]})"

    def result(self):
        val = self.get_value(self.x)
        return np.abs(val)

    def get_grad(self, grad):
        val = (self.x.x != self.val)
        val = np.where(val == 0, np.array(1.), np.array(-1.))
        return check_shape(grad * val, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, self.grad_name)
        return self.holder * Abs(self.x),


class Prime:

    def __init__(self,
                 func,
                 grad,
                 create_graph=False,
                 grad_name=""):
        self.grad = grad
        self.grad_name = grad_name
        self.create_graph = create_graph
        self.prime(func)

    def prime(self, func):
        __stack = [(func, self.grad)]
        create_graph = self.create_graph
        append = __stack.append
        while __stack:
            current_func, grad = __stack.pop()
            if isinstance(current_func, (int, float, np.ndarray, numpy.ndarray)):
                continue
            var = current_func.vars
            if var == 0 or var == 3:
                if create_graph:
                    current_func.graph.grad_name = self.grad_name
                current_func.grad = grad if current_func.grad is None else current_func.grad + grad
                if var == 3:
                    current_func.parent.cumulate += 1
                    if current_func.parent.cumulate == func.has_conv:
                        append((current_func.parent, None))

            elif var == 2:
                grad1, grad2 = current_func.get_grad(grad)
                if create_graph:
                    if current_func.graph:
                        current_func.holder = current_func.graph
                        graphs = current_func.get_graph()
                        current_func.graph = None
                        current_func.holder = None
                    else:  # first node
                        current_func.holder = self.grad
                        graphs = current_func.get_graph()
                    if hasattr(current_func.x, 'graph'):
                        current_func.x.graph = graphs[0] if current_func.x.graph is None else current_func.x.graph + graphs[0]
                        current_func.x.graph.parent = current_func.x
                        current_func.x.son = current_func.x.graph
                    if hasattr(current_func.y, 'graph'):
                        current_func.y.graph = graphs[1] if current_func.y.graph is None else current_func.y.graph + graphs[1]
                        current_func.y.graph.parent = current_func.y
                        current_func.y.son = current_func.y.graph
                append((current_func.x, grad1))
                append((current_func.y, grad2))

            elif var == 1:
                grad = current_func.get_grad(grad)
                if create_graph:
                    if current_func.graph:
                        current_func.holder = current_func.graph
                        graphs = current_func.get_graph()
                        current_func.graph = None
                        current_func.holder = None
                    else:  # first node
                        current_func.holder = self.grad
                        current_func.grad_name = self.grad_name
                        graphs = current_func.get_graph()
                    if hasattr(current_func.x, 'graph'):
                        current_func.x.graph = graphs[0] if current_func.x.graph is None else current_func.x.graph + graphs[0]
                        current_func.x.graph.parent = current_func.x
                        current_func.x.son = current_func.x.graph
                append((current_func.x, grad))

            elif var == 4:
                grad = current_func.get_grad(grad)
                current_func.cumulate = 0
                if create_graph:
                    if current_func.graph:
                        current_func.holder = current_func.graph
                        graphs = current_func.get_graph()
                        current_func.graph = None
                        current_func.holder = None
                    else:  # first node
                        current_func.holder = [i.graph if i.graph is not None else Matrix(np.zeros(i.shape), "0") for i in current_func.x3]
                        for i in current_func.x3:
                            i.graph = None
                        current_func.grad_name = self.grad_name
                        graphs = current_func.get_graph()
                        graphs[0].has_conv = len(current_func.x3)
                    if hasattr(current_func.x, 'graph'):
                        current_func.x.graph = graphs[0] if current_func.x.graph is None else current_func.x.graph + graphs[0]
                        current_func.x.graph.parent = current_func.x
                        current_func.x.son = current_func.x.graph
                append((current_func.x, grad))


class View:

    def __init__(self,
                 func,
                 total_task,
                 graph_object=None,
                 head=False,
                 parent=None,
                 shape='box',
                 filename='view.json',
                 time_begin=0.,
                 func_head=None,
                 grad_name="",):
        self.graph_object = graph_object
        self.func_head = func_head
        if head:
            ls = filename.split(".")
            self.graph_object = graphviz.Digraph('g',
                                                 filename=ls[0],
                                                 strict=True,
                                                 format=ls[1])
            self.graph_object.attr('node', shape=shape)
            self.func_head = func
            parent = str(id(func))
        self.time = time_begin
        self.parent = parent
        self.total = total_task
        self.head = head
        self.grad_name = grad_name
        self.current_task = 0
        self.current_task = self.view(func)

    def view(self, func):
        current_task = 0
        total = self.total
        begin = self.time
        _stack = [(func, 0)]
        append = _stack.append
        pop = _stack.pop
        equalSlice = 0
        while _stack:
            current_task += 1
            current_func, idx = pop()
            if isinstance(current_func, (int, float, np.ndarray)):

                self.print_result(current_task, total, begin)
                self.graph_object.node(f"{id(current_func)}_{idx}",
                                       f"{current_func}\n()",
                                       style='filled',
                                       fillcolor='0 1 1')

            elif current_func.vars == 5:
                self.print_result(current_task, total, begin)
                string_func = f"{id(current_func)}_{idx}"
                label = current_func.label
                # create node
                self.graph_object.node(name=string_func,
                                       label=current_func.__class__.__name__,
                                       style='filled',
                                       fillcolor=color_map[type(current_func)])

                for index, part in enumerate(current_func.x):
                    append((part, current_task))
                    if current_task > 1:  # head
                        current_func.x1 = equation, shape = self.get_label_shape1(part)
                        self.graph_object.edge(f"{id(part)}_{current_task}", string_func,
                                               label=f"<stack{index} = {str(equation)}<BR/>{shape}>")
                    else:
                        optimize = str(current_func.label)
                        self.graph_object.node(name=str(id(label)), label=f"<{optimize}<BR/>{current_func.val.shape}>", style='filled',
                                               fillcolor=color_map[type(current_func)])
                        self.graph_object.edge(string_func, str(id(label)))
                        current_func.x1 = equation, shape = self.get_label_shape1(part)
                        self.graph_object.edge(f"{id(part)}_{current_task}", string_func,
                                               label=f"<{index}: {str(equation)}<BR/>{shape}>")

            elif current_func.vars == 3:
                self.print_result(current_task, total, begin)
                self.graph_object.node(f"{id(current_func)}_{idx}",
                                       f"{current_func.label}\n{current_func.shape}",
                                       style='filled',
                                       fillcolor='0 1 1')
                self.graph_object.edge(f"{id(current_func.parent)}_{equalSlice}",
                                       f"{id(current_func)}_{idx}",
                                       label=f"<{str(current_func.label)}<BR/>{current_func.shape}>")
                append((current_func.parent, current_task))

            elif current_func.vars == 0:
                self.print_result(current_task, total, begin)
                self.graph_object.node(f"{id(current_func)}_{idx}",
                                       f"{current_func.label}\n{current_func.shape}",
                                       style='filled',
                                       fillcolor='0 1 1')
                if current_task == 1:
                    self.graph_object.edge(f"{id(current_func)}_{idx}", f"{id(current_func)}_{idx}",
                                           label=f"<{current_func.label}<BR/>{current_func.shape}>")

            elif current_func.vars == 2:
                self.print_result(current_task, total, begin)

                string_func = f"{id(current_func)}_{idx}"
                label = current_func.label
                # create node
                self.graph_object.node(name=string_func,
                                       label=current_func.__class__.__name__,
                                       style='filled',
                                       fillcolor=color_map[type(current_func)])
                append((current_func.x, current_task))
                append((current_func.y, current_task))

                if current_task > 1:
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    current_func.x2 = equation2, y_shape = self.get_label_shape1(current_func.y)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{str(equation1)}<BR/>{x_shape}>")
                    self.graph_object.edge(f"{id(current_func.y)}_{current_task}", string_func, label=f"<{str(equation2)}<BR/>{y_shape}>")
                else:
                    optimize = str(current_func.label)
                    self.graph_object.node(name=str(id(label)), label=f"<{optimize}<BR/>{current_func.shape}>", style='filled', fillcolor=color_map[type(current_func)])
                    self.graph_object.edge(string_func, str(id(label)))
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    current_func.x2 = equation2, y_shape = self.get_label_shape1(current_func.y)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{str(equation1)}<BR/>{x_shape}>")
                    self.graph_object.edge(f"{id(current_func.y)}_{current_task}", string_func, label=f"<{str(equation2)}<BR/>{y_shape}>")

            elif current_func.vars == 1:
                self.print_result(current_task, total, begin)
                string_func = f"{id(current_func)}_{idx}"

                self.graph_object.node(name=string_func,
                                       label=current_func.__class__.__name__,
                                       style='filled',
                                       fillcolor=color_map[type(current_func)])
                append((current_func.x, current_task))

                if current_task > 1:
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{str(equation1)}<BR/>{x_shape}>")
                else:
                    optimize = str(current_func.label)
                    label = str(id(current_func.label))
                    self.graph_object.node(name=label, label=f"<{optimize}<BR/>{current_func.x.shape}>", style='filled', fillcolor=color_map[type(current_func)])
                    self.graph_object.edge(string_func, label)
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{str(equation1)}<BR/>{x_shape}>")

            elif current_func.vars == 4:  # EqualSlice
                self.print_result(current_task, total, begin)

                string_func = f"{id(current_func)}_{equalSlice}"
                equalSlice += 1
                self.graph_object.node(name=string_func,
                                       label=current_func.__class__.__name__,
                                       style='filled',
                                       fillcolor=color_map[type(current_func)])
                append((current_func.x, current_task))
                if current_task > 1:
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{equation1}<BR/>{x_shape}>")
                else:
                    optimize = str(current_func.label)
                    label = str(id(current_func.label))
                    self.graph_object.node(name=label, label=f"<{optimize}<BR/>{current_func.shape}>", style='filled', fillcolor=color_map[type(current_func)])
                    self.graph_object.edge(string_func, label)
                    current_func.x1 = equation1, x_shape = self.get_label_shape1(current_func.x)
                    self.graph_object.edge(f"{id(current_func.x)}_{current_task}", string_func, label=f"<{equation1}<BR/>{x_shape}>")

            elif isinstance(current_func.x, (int, float, np.ndarray)):
                self.print_result(current_task, total, begin)
        return current_task

    @staticmethod
    def get_label_shape1(current_func: int | float | np.ndarray | Function):
        equation = current_func.label if hasattr(current_func, 'label') else str(current_func)
        shape = current_func.shape if hasattr(current_func, "shape") else current_func
        return equation, shape


    @staticmethod
    def print_result(current, total, begin):
        lis = ['[' if i == 0 else ']' if i == 21 else ' ' for i in range(22)]
        index = int((current + 1) / total * 20)
        percentage = format(current * 100 / total, '.2f')
        if 0 <= index < 20:
            pass
        else:
            index = 20
        if index > 0:
            for i in range(1, index + 1):
                lis[i] = u'\u25A0'
            string = ''.join(lis)
            time1 = time() - begin
            print(f'\r{string} {percentage}% Time: {time1:.3f}s',
                  end='',
                  flush=True)
        else:
            string = ''.join(lis)
            time1 = time() - begin
            print(f'\r{string} {percentage}% Time: {time1:.3f}s',
                  end='',
                  flush=True)


def set_grads(x, y):
    if hasattr(x, 'grad'):
        x.grad = Matrix(x.grad)
    if hasattr(y, 'grad'):
        y.grad = Matrix(y.grad)


def check_shape(grad, inp):
    """
    :return: real grad
    """
    try:
        shape = inp.shape
    except:
        shape = ()
    try:
        grad_shape = grad.shape
    except:
        grad_shape = ()
    offset = len(grad_shape) - len(shape)
    if offset == 0 and grad_shape == shape:
        return grad
    elif offset < 0:
        raise RuntimeError(f"grad shape {grad_shape} smaller than {shape}")
    elif offset == 0 and grad_shape != shape:
        repeats1 = [max(s1, s2) // min(s1, s2) for s1, s2 in zip(shape, grad_shape)]
        grad = np.sum(grad, axis=np.argmax(repeats1), keepdims=True)
    for _ in range(offset):
        grad = np.sum(grad, axis=0)
    if shape and np.size(grad) != np.size(inp.val) and not np.size(grad) % np.size(inp.val):
        grad = np.sum(grad, axis=0, keepdims=True)
    return grad


def graph_check_shape(grad, inp):
    """
    :return: real grad
    """
    try:
        shape = inp.shape
    except:
        shape = ()
    try:
        grad_shape = grad.shape
    except:
        grad_shape = ()
    offset = len(grad_shape) - len(shape)
    if offset == 0 and grad_shape == shape:
        return grad
    elif offset < 0:
        raise RuntimeError(f"grad shape {grad_shape} smaller than {shape}")
    elif offset == 0 and grad_shape != shape:
        repeats1 = [max(s1, s2) // min(s1, s2) for s1, s2 in zip(shape, grad_shape)]
        grad = sum(grad, axis=np.argmax(repeats1), keepdims=True)
    for _ in range(offset):
        grad = sum(grad, axis=0, keepdims=True)
    if shape and np.size(grad.val) != np.size(inp.val) and not np.size(grad.val) % np.size(inp.val):
        grad = sum(grad, axis=0, keepdims=True)
    return grad


def reset_graph(*var):
    graphs = [i.graph for i in var]
    for i in var:
        i.grad = None
        i.graph = None
    return graphs


def change_shape(x, colms, index):
    temp = x[index]
    x[index] = temp // colms
    x.insert(len(x) + 1 + index, colms)
    return x


def get_label(*args):
    ls = []
    for i in args:
        if hasattr(i, 'label'):
            i.get_label()
            ls.append(i.label)
        else:
            ls.append("array")
    return ls


def search_cnodes(func, count=0):
    if func.vars == 2:
        count = search_nodes(func.x, count)
        count = search_nodes(func.y, count)
    elif func.vars == 1:
        count = search_nodes(func.x, count)
    elif func.vars == 3:
        count += 1
    return count


def search_nodes(func, count=0):
    stack = [func]

    while stack:
        current_func = stack.pop()
        if isinstance(current_func, (int, float, np.ndarray, list)):
            count += 1
        elif current_func.vars == 5:
            current_func.get_label()
            count += 1
            for i in current_func.x:
                stack.append(i)
        elif current_func.vars == 0 or current_func.vars == 3:
            current_func.get_label()
            count += 1
            if current_func.vars == 3:
                stack.append(current_func.parent)
        elif current_func.vars == 2:
            current_func.get_label()
            count += 1
            stack.append(current_func.x)
            stack.append(current_func.y)
        elif current_func.vars == 1:
            current_func.get_label()
            count += 1
            stack.append(current_func.x)
        elif current_func.vars == 4:
            current_func.get_label()
            count += 1
            current_func.cumulate = 0
            stack.append(current_func.x)
        else:
            count += 1
    return count


def replace_upscript(inp: str):
    string = inp.split("**")
    left = 0
    right = 0
    FLAG = False
    ls = [list(i) for i in string]
    digit_mode = False
    op = ''
    for index, i in enumerate(ls):
        i.append('<SUP>')
    ls[-1].pop()
    for index, i in enumerate(ls):
        if index > 0:
            for idx, word in enumerate(i):
                if not FLAG and word.isdigit():
                    digit_mode = True
                if idx - len(i) == -1 and digit_mode:
                    i.append('</SUP>')
                    break
                if digit_mode and not word.isdigit():
                    i.insert(idx, '</SUP>')
                    digit_mode = False
                    FLAG = True
                    right = 0
                    left = 0
                if not digit_mode and not FLAG and i == '(':
                    left += 1
                if not digit_mode and not FLAG and i == ')':
                    right += 1
                if left > 0 and left == right:
                    left = 0
                    right = 0
                    i.insert(idx, '</SUP>')
            FLAG = False
        op += ''.join(i)
    return op


def check_brocast(result1, result2):
    try:
        broadcasted = np.broadcast(result1, result2)
        shape = broadcasted.shape
        repeats1 = max([s1 // s2 if s1 % s2 == 0 else -1 for s1, s2 in zip(shape, result2.shape)])
        repeats2 = max([s1 // s2 if s1 % s2 == 0 else -1 for s1, s2 in zip(shape, result1.shape)])
        return max(repeats1, repeats2)
    except ValueError:
        return 0

color_map = {
    Divide:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Add:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Multi:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sub:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Matmul:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    starmulti:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    exp:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sin:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    tan:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sec:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    pow:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    ln:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcsin:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcos:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcot:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arctan:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    cos:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    csc:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    cot:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sqrt:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    square:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    transpose:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    trace:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    inv:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sum:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Max:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Slice:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    reshape:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Abs:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Min:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    mean:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    log:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    repeat:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    tanh:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    EqualSlice:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Sigmoid:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    ScalarToMatrix:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    stack:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    softmax:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    tensordot:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
}
