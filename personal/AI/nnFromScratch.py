import matplotlib.pyplot as plt
from array import array
from multiprocessing.dummy import Array
from turtle import forward
import numpy as np 
def relu(x):
    return np.maximum(0,x)
def step(x):
    if type(x) == list:
        out = []
        for i in x:
            if i > 0:
                out.append(1)
            else:
                out.append(0)

        return out
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
        self.weights = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the weights. w[x][:] corresponds to output[x], w[:][x] corresponds to input[x]
        self.weightError = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the error of each of the weights
        self.valsError = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the error of each of the nodes with respect to each other node. valsError[a][b] means the error of the ath node versus the bth connection

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

    def func2(self, targets, x, count): #for node errors
        
        if self.type == "output": #derivitive for output layer 
            arr = [] 
            for i in range(len(targets)):
                arr.append(2*(targets[i] - x[i])) #derivative of (t-x)^2 = 2(t-x) 
            #print("OUT ARR", arr)
            return(arr)
        
        else:   #node derivative 
            arr = []            
            #print("____")
            #print(")))")
            #print(np.expand_dims(self.valsError[count],1))
            
            functionOut = np.array(self.connection.func2(targets, self.connection.vals, 0)) #variable to store the result of function
            error = np.array(self.valsError) #make into a np array
            error = error.T #transpose the array so it is formatted correctly 
           
            arr.append(error.dot(functionOut)) #how much does the node effect each of the next layers   
            
            
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
                arr2.append(error[i][g]*(functionOut[g][0])) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) 9self.weighterror * f'(vals)
            arr.append(arr2)
    
        return(arr)

        ## w1 corresponds to l1, need to average deriv for l1 (l1- o1 and l1 - o2)
        ## add np.dot() to add up all the derivitives and then divide ? 
    
    
    def update(self, targets, lr): #start of backpropogation
        arr = self.func3(targets) #get errors of weights
        #print("upd arr", arr)

        for i in range(len(arr)):
        
            for p in range(len(arr[0])):
               self.weights[i][p] += lr*arr[i][p] #update weights by error*lr 
              
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
input = dense(2)
dense1 = dense(3)
dense2 = dense(3)
out = output(2)
print("set")

input.connect(dense1)
dense1.connect(dense2)
dense2.connect(out)
print("set")
ins = np.random.rand(1000,2)
outs = []
p = [3,-2,5,5,0,-3,-3]
print(step(p))
for i in range(len(ins)):
    ins[i][0] *= 1
    ins[i][1] *= 1
c1 = 0 
c2 = 0
for i in ins:

    if i[1] > (i[0]*2.2-0.5)**2:
        outs.append([0,1])
        c1 += 1
    else:
        outs.append([1,0])
        c2 += 1
print(c1,c2)
mses = [0,0]
lr = 0.0001
epochs = 100
aes = [0,0] 
for x in range(epochs):
    for i in range(len(ins)):

        #input.input([1,2,2])
        input.input(ins[i])
        input.forward()
        dense1.forward()
        dense2.forward()
        input.backward()
        dense1.backward()
        dense2.backward()


        input.update(outs[i], lr)
        dense1.update(outs[i], lr)
        dense2.update(outs[i], lr)

        input.input(ins[i])
        input.forward()
        dense1.forward()
        dense2.forward()
        mses[0] += float(out.forward(outs[i])[0][0])
        mses[1] += float(out.forward(outs[i])[0][1])
        #aes[0] += float(out.forward(outs[i])[1][0])
        
        #aes[1] += float(out.forward(outs[i])[1][1])
        out1 = float(out.forward(outs[i])[1][1])
        out2 = float(out.forward(outs[i])[1][0])
        if out1 > out2 and i <10:
            #print(1)
            #print(out1-out2)
            pass
        elif i < 10: 
            pass
            #print(100000)
            #print(out1-out2)        
mses[0] = mses[0]/(len(ins)*epochs)
mses[1] = mses[1]/(len(ins)*epochs)

print(mses)
print("Ae =", aes[0]/len(ins)*epochs, aes[1]/len(ins)*epochs)

ins = np.random.rand(300,2)
outs = []

for i in range(len(ins)):
    ins[i][0] *= 2
    ins[i][1] *= 1
for i in ins:

    if i[1] > (i[0]*2.2-0.5)**2:
        outs.append([0,1])
        c1 += 1
    else:
        outs.append([1,0])
        c2 += 1
print(c1,c2)
mses = [0,0]
ae = [0,0]
reds = [] 
blues = [] 
for i in range(len(ins)):

    #input.input([1,2,2])
    input.input(ins[i])
    input.forward()
    dense1.forward()
    dense2.forward()
    result1 = float(out.forward(outs[i])[0][0])
    result2 = float(out.forward(outs[i])[0][1])
    result3 = float(out.forward(outs[i])[1][0])
    result4 = float(out.forward(outs[i])[1][1])
    if result3>result4:
        reds.append(ins[i].tolist())
    else:
        blues.append(ins[i].tolist())
    mses[0] += result1
    mses[1] += result2
    """
    print("input = ", ins[i])
    print("targets = ", outs[i])
    print("out1 = ",result3)
    print("out2 = ",result4)
    print("ms1 = ", result1)
    print("ms2 = ", result2 )
    """
print("raw mses", mses)
mses[0] = mses[0]/(len(ins))
mses[1] = mses[1]/(len(ins))
print(mses)
print(len(reds))
print(len(blues))
       
plt.scatter(reds[:][0],reds[:][1])
plt.scatter(blues[:][0],blues[:][1])
plt.show()

        

