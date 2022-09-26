import numpy as np 
def relu(x):
    return np.maximum(0,x)
def step(x):
    return
a = np.array([2,2])
b = np.array([[2,2],[2,2],[2,2]])
c = b.dot(a)
print(np.zeros((2,3)))
print(c)
class dense():
    def __init__(self,nodes):
        self.nodes = nodes
        self.noActVals = np.array(nodes)
        self.vals = np.array(nodes)
        self.valsError = np.array(nodes)
    def connect(self, connection):
        self.weights = np.zeros((connection.nodes,self.nodes))
        self.weightError = np.zeros((connection.nodes,self.nodes))
    def forward(self,connection):
        connection.noActVals = self.weights.dot(self.vals)
        connection.vals = relu(connection.noActVals)
    def backward(self, connection ):
        for i in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                self.weightError[i][x] = self.vals[i]
                self.weightError[i][x] *= "step function"(connection.vals[x])
        for i in range(len(self.vals)):
            self.valsError[i] = 0 
            for x in range(len(connection.vals)):
                self.valsError[i] += self.weights[i][x]
            

"""



"""