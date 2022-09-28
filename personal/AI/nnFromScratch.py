
import numpy as np 
def relu(x):
    return np.maximum(0,x)
def step(x):
    out = x > 0
    ans = x[:]
    for i in range(len(out)):
        ans[i] = int(out[i])

    return ans 

a = np.array([2,2])
b = np.array([[2,2],[2,2],[2,2]])
c = b.dot(a)
print(np.zeros((2,3)))
print(c)
class dense():
    def __init__(self,nodes, connection):
        self.nodes = nodes
        self.noActVals = np.array(nodes)
        self.vals = np.array(nodes)
        self.type = "hidden"
        self.connection = connection

    def connect(self):
        self.weights = np.zeros((self.connection.nodes,self.nodes))
        self.weightError = np.zeros((self.connection.nodes,self.nodes))
        self.valsError = np.zeros((self.connection.nodes,self.nodes))
    def forward(self):
        self.connection.noActVals = self.weights.dot(self.vals)
        self.connection.vals = relu(self.connection.noActVals)

    def backward(self ):
        for i in range(len(self.weights)):
            for x in range(len(self.weights[0])):
                self.weightError[i][x] = self.vals[i]
                self.weightError[i][x] *= step(self.connection.vals[x])
        for i in range(len(self.vals)):
            self.valsError[i] = 0 
            for x in range(len(self.connection.vals)):
                self.valsError[i][x] = self.weights[i][x]*step(self.connection.vals[x])
    
    def func(x,y):
        temp = 0 
        for i in y:
         temp += x*i
        temp = step(temp)
        temp = temp*x
        return temp

    def func2(self, targets, x):
        if self.type == "output":
            arr = [] 
            for i in range(len(targets)):
                arr.append(2*(targets[i] - x[i]))
            return(arr)
        else:
            arr = [] 
            for i in range(len(self.valsError)):
                arr2 = [] 
                for p in range(len(self.valsErro[0])):
                    arr2.append(self.valsError[i][p]*self.connection.func2(targets, self.connection.vals))
                arr.append(arr2)
            return(arr)        
    
    def func3(self, targets, x):
        arr = [] 
        for i in range(len(self.weights)):
            arr2 = []
            for p in range(len(self.weights[0])):
                arr2.append(self.vals[p].dot(self.connection.func2(targets, self.connection.vals)))
                arr2[p] = int(arr[p].dot([1/len(arr[p])]))

            arr.append(arr2)
        return(arr)

        ## w1 corresponds to l1, need to average deriv for l1 (l1- o1 and l1 - o2)
        ## add np.dot() to add up all the derivitives and then divide ? 
    
    
    def update(self, targets, x, lr):
        arr = self.func3(targets, x)
        for i in range(len(arr)):
            for p in range(len(arr)):
                self.weights[i][p] -= lr*arr[i][p]    
class output(dense):
    def forward():
        pass 
        
"""
o = max(0, l1*w1 + l2*w2 etc...)


"""


        

