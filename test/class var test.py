class parent:

    def __init__(self, var):
        self.var = var

class child(parent):

    def __init__(self, var):
        super().__init__()



test1 = parent()

parent.var = 2

test2 = parent()
print(test1.var)