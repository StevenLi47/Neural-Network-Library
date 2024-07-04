import copy
import numpy as np
from Layers import activation as ac, GradDescent as gd
from Model import sequence as seq

class dense(seq.node):

    def __init__(self, in_shape, descent, outputs, activ = None):
        super().__init__()
        inputs = in_shape[0]
        if isinstance(activ, ac.relu):
            scale = np.sqrt(2 / inputs)
        elif isinstance(activ, ac.tanh) or isinstance(activ, ac.sigmoid):
            scale = np.sqrt(2 / (inputs + outputs))
        else:
            scale = 0.1

        self.weight = np.random.normal(loc=0, scale=scale, size=(outputs, inputs))
        self.bias = np.random.normal(loc=0, scale=scale, size=(outputs, 1))
        self.W_descent = copy.deepcopy(descent)
        self.b_descent = copy.deepcopy(descent)

    def forward(self, input):
        self.input = input
        return np.matmul(self.weight, self.input) + self.bias

    def backward(self, grad, lr):
        bias_grad = np.mean(grad, axis=1).reshape(-1, 1)
        self.bias -= lr * self.W_descent.backwards(bias_grad)
        weight_grad = np.matmul(grad, self.input.T)
        self.weight -= lr * self.b_descent.backwards(weight_grad)
        return np.matmul(self.weight.T, grad)


class dropout(seq.node):

    def __init__(self, in_shape, rate):
        super().__init__()
        self.in_shape = in_shape
        self.rate = rate

    def forward(self, input):
        self.drop_out = np.random.choice([0, 1], self.in_shape, p = [1 - self.rate, self.rate]).astype(float)
        output = np.multiply(self.drop_out, input)
        return output

    def backward(self, grad, lr):
        return np.multiply(self.drop_out, grad) / self.rate

sequence = seq.sequence()
sequence.add(dense(4, 4, gd.adam, activ = ac.relu))
sequence.add(dense(4, 4, gd.adam, activ = ac.relu))
sequence.add(dense(4, 4, gd.adam, activ = ac.relu))