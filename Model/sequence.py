import numpy as np
import copy
import error as er

class node():

    def __init__(self):
        self.next = None
        self.prev = None

    def forward(self, input):
        return input

    def backward(self, grad, lr):
        return grad

    def setNext(self, next_node):
        self.next = next_node
        next_node.prev = self

    def setPrev(self, prev_node):
        self.prev = prev_node
        prev_node.next = self


class sequence():

    def __init__(self, in_shape):
        self.frame = np.zeros(in_shape)
        self.buffer = node()
        self.current = self.buffer
        self.length = 0
        self.position = -1
        self.forward = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == -1:
            self.current = self.buffer

        if self.forward:
            self.current = self.current.next
            self.position += 1
        else:
            self.current = self.current.prev
            self.position += 1

        if self.position != len(self):
            return self.current
        else:
            self.position = -1
            raise StopIteration

    def __getitem__(self, item):
        self.current = self.buffer
        self.position = -1

        if isinstance(item, int):
            for i in range(item + 1):
                self.current = self.current.next

            return self.current

        elif isinstance(item, slice):
            start, end, step = item.indices(len(self))
            sub_seq = sequence()

            for i in range(start + 1):
                self.current = self.current.next
                self.position += 1

            while self.position + step <= end:
                node_copy = copy.copy(self.current)
                node_copy.next, node_copy.prev = (None, None)
                sub_seq.add(node_copy)

                for i in range(step):
                    self.current = self.current.next
                    self.position += 1

            return sub_seq

    def __len__(self):
        return self.length

    def __reversed__(self):
        seq_copy = copy.copy(self)
        seq_copy.change_direct()
        return seq_copy

    def getShape(self):
        return np.shape(self.frame)

    def change_direct(self):
        if self.forward:
            self.forward = False
        else:
            self.forward = True

    def add(self, node):
        self.current.setNext(node)
        self.current = node
        self.current.setNext(self.buffer)
        self.length += 1
        self.frame = self.current.forward(self.frame)
        node.in_shape = self.getShape()

    def pop(self):
        self.buffer.setPrev(self.buffer.prev.prev)
        self.length -= 1

    def full_forward(self, input):
        for node in self:
            input = node.forward(input)

        return input

    def full_backward(self, grad, lr):
        self.change_direct()
        for node in self:
            grad = node.backward(grad, lr)
        self.change_direct()

    def train(self, input, output, epoch, lr, error = er.mse):
        for e in range(epoch):
            guess = self.full_forward(input)
            print("epoch:", e + 1)
            print("error: {0:.4f}".format(error.forward(guess, output)))
            print(*np.round(guess, 4)[0], "\n")

            self.full_backward(error.backward(guess, output), lr.learning_rate)
            lr.update(e)

        guess = self.full_forward(input)
        print("error: {0:.4f}".format(error.forward(guess, output)))
        print(*np.round(guess, 4)[0], "\n")

    def inference(self, input):
        guess = self.full_forward(input)
        print(*np.round(guess, 4)[0], "\n")


class dataset():

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.dataset = zip(self.inputs, self.outputs)