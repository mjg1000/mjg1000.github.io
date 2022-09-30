
from array import array
from multiprocessing.dummy import Array
from turtle import forward
import numpy as np 
def relu(x):
    return np.maximum(0,x)
def step(x):
    if type(x) == Array:
        out = x > 0

        ans = x[:]
        for i in range(len(out)):
            ans[i] = int(out[i])

        return ans 
    else:
        if x > 0:
            return(1)
        else:
            return 0 


a = np.array([2,2])
b = np.array([[2,2],[2,2],[2,2]])
c = b.dot(a)
print(np.zeros((2,3)))
print(c)
class dense():
    def __init__(self,nodes):
        self.nodes = nodes #number of nodes in the layer 
        self.noActVals = np.array(nodes) #the value of each node, without being activated
        self.vals = np.array(nodes) #the value of each node
        self.type = "hidden" #if this layer is an output 
    def input(self, values):
        self.vals = values #set each value for the input layer 

    def connect(self, connection): #link to the next layer 
        self.connection = connection #the next layer 
        self.weights = np.random.rand(self.connection.nodes,self.nodes) #the weights. w[x][:] corresponds to output[x], w[:][x] corresponds to input[x]
        self.weightError = np.zeros((self.connection.nodes,self.nodes)) #the error of each of the weights
        self.valsError = np.zeros((self.connection.nodes,self.nodes)) #the error of each of the nodes with respect to each other node. valsError[a][b] means the error of the ath node versus the bth connection

    def forward(self): #compute the next values 
        self.connection.noActVals = self.weights.dot(self.vals) #dot product of weights and values
        self.connection.vals = relu(self.connection.noActVals) #activating 

    def backward(self ):

        for i in range(len(self.vals)):
            for x in range(len(self.connection.vals)): #iterate through each weight 
                self.weightError[x][i] = self.vals[i] #weight error = (g'(x) * f'(g(x)) g(x) = weight*node so g'(x) = node. 
                self.weightError[x][i] *= step(self.connection.vals[x]) #f(x) = relu so f'(x) = step, so multiplying by step(g(x))
        
        for i in range(len(self.vals)): #iterate through the node errors 
            for x in range(len(self.connection.vals)):
                self.valsError[x][i] = self.weights[x][i]*step(self.connection.vals[x]) # g(x) = node*weight so g'(x) = weight. step of the connection  valsError[x] corresponds to output[x]
    def func(x,y):
        temp = 0 
        
        for i in y:
         temp += x*i
        
        temp = step(temp)
        temp = temp*x
        return temp

    def func2(self, targets, x, count): #for node errors
        
        if self.type == "output": #derivitive for output layer 
            arr = [] 
            print(x)
            for i in range(len(targets)):
                arr.append(2*(targets[i] - x[i]))
            #print("OUT ARR", arr)
            return(arr)
        
        else:   #node derivative 
            arr = []            
            #print("____")
            #print(")))")
            #print(np.expand_dims(self.valsError[count],1))
            
            functionOut = np.array(self.connection.func2(targets, self.connection.vals, 0))
            error = np.array(self.valsError)
            #error = np.array([[error],[error]])
            error = error.T
            #error = error.squeeze(1)
            #print("e1", self.valsError)
            #print("nodes", self.nodes)
            #print("error", error)
            #print("function", functionOut)
            #arr.append(np.array(self.connection.func2(targets, self.connection.vals, p)).dot(np.expand_dims(self.valsError,1))) #how much does the node effect each of the next layers
            arr.append(error.dot(functionOut)) #how much does the node effect each of the next layers
            #print(arr)
            #arr[p] = np.expand_dims(np.array(arr[p]),1).dot(np.array([1/len(arr[p])]))
            #print(np.array(arr))
            
            
            for g in range(len(arr)):
                arr[g] = arr[g]/len(functionOut)
            
            #print("should be 1 num")
            
            arr = np.array(arr)
            #print(arr)
            return(arr.squeeze(0))        
    
    def func3(self, targets): #backpropogation algo
        
        arr = [] #instantiate array (size = self.weights)
        # SHOULD BE DONE
        functionOut = np.expand_dims(np.array(self.func2(targets, self.connection.vals, 0 )),1)
        for i in range(len(self.weights)): #for each val error 
            error = np.array(self.weightError)
            arr2 = []
            for g in range(len(self.weightError[0])):    
        
                #arr2.append(np.expand_dims(np.array(self.weightError[:][i]),1).dot(self.func2(targets, self.connection.vals[i], i ))) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) 9self.weighterror * f'(vals)
                
                #fOutSlice = np.array(functionOut[i])
                #print("fout = ", functionOut)
                #print("weight err",np.expand_dims(np.array(self.weightError[i]),1))
                #arr.append(functionOut.dot(np.expand_dims(np.array(self.weightError[i]),1).T)[0]) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) 9self.weighterror * f'(vals)
                #print("func3 err", error[i][g])
                #print("f3 ws", self.weightError)
                #print("func3", functionOut[g])
                arr2.append(error[i][g]*(functionOut[g][0])) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) 9self.weighterror * f'(vals)
                #print("f3 arr", arr2)
                #arr2[p] = int(np.array(arr[p]).dot([1/len(arr[p])]))
                #print(np.expand_dims(np.array(self.weightError[i]),1))
                #print("arr", arr)
            arr.append(arr2)
    
        return(arr)

        ## w1 corresponds to l1, need to average deriv for l1 (l1- o1 and l1 - o2)
        ## add np.dot() to add up all the derivitives and then divide ? 
    
    
    def update(self, targets, lr): #start of backpropogation
        arr = self.func3(targets) 
        #print("upd arr", arr)

        for i in range(len(arr)):
        
            for p in range(len(arr[0])):
                self.weights[i][p] += lr*arr[i][p]    

class output(dense):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.type = "output"
    
    def forward(self, targets):
        arr = [] 
        for i in range(len(self.vals)):
            arr.append((targets[i]-self.vals[i])**2)
        return(arr,self.vals)

#NETWORK BEGINS:
input = dense(3)
dense1 = dense(4)
dense2 = dense(6)
out = output(2)
print("set")

input.connect(dense1)
dense1.connect(dense2)
dense2.connect(out)
print("set")
input.input([1,2,2])
input.forward()
dense1.forward()
dense2.forward()
print(out.forward([0,0,0]))
input.backward()
dense1.backward()
dense2.backward()


input.update([0,0], 0.0001)
dense1.update([0,0],0.0001)
dense2.update([0,0],0.0001)

input.input([1,2,2])
input.forward()
dense1.forward()
dense2.forward()
print("_______________________________________________________")
print(out.forward([0,0,0]))
"""
o = max(0, l1*w1 + l2*w2 etc...)


"""


        

        

