import numpy as np
import copy
import error as er

class layer():

    def __init__(self, agg = lambda x: sum(x)):
        self.forwards = []
        self.backwards = []
        self.aggregation = agg

    def setPointers(self, *layers, cond = None):
        for layer in layers:
            point = pointer(self, layer, cond)
            self.forwards.append(point)
            layer.backwards.append(point)

    def forward(self, *inputs):
        self.aggregation([inputs])

    def backward(self, lr, *grad):
        self.aggregation([grad])


class pointer():

    def __init__(self, forward, backward, cond = None):
        self.forward = forward
        self.backward = backward
        self.condition = cond
        self.input = None

    def setInput(self, input):
        self.input = input

    def forward(self, *inputs):
        self.input = self.forward.forward(inputs)

    def backward(self, lr, func = lambda x: sum(x), *grads):
        self.combined_grad = func([grads])
        return self.parent.backward(self.combined_grad, lr)


class network():

    def __init__(self):
        self.forward = True

    def add(self, layer, ):