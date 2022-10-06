import matplotlib.pyplot as plt
from array import array
from multiprocessing.dummy import Array
from turtle import forward
import numpy as np 
import math
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
def sigmoid(x):
    return ( 1/(1+math.e**(-x)))
def sigmoidDeriv(x):
    return (sigmoid(x)*(1-sigmoid(x)))



a = np.array([3,4])
b = np.array([[2,2],[3,3],[4,4]])
c = b.dot(a)
print(np.zeros((2,3)))
print(c)
class dense():
    def __init__(self,nodes):
        self.nodes = nodes #number of nodes in the layer 
        self.noActVals = np.array(nodes) #the value of each node, without being activated
        self.vals = np.array(nodes) #the value of each node
        self.type = "hidden" #if this layer is an output 
        self.updateArr1 = []
        self.updateArr2 = []
        self.updateArr3 = []
        self.updateArr4 = []
    
    def input(self, values):
        self.vals = values #set each value for the input layer 

    def connect(self, connection): #link to the next layer 
        self.connection = connection #the next layer 
        self.weights = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the weights. w[x][:] corresponds to output[x], w[:][x] corresponds to input[x]
        self.weightError = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the error of each of the weights
        self.valsError = np.random.rand(self.connection.nodes,self.nodes)*2-1 #the error of each of the nodes with respect to each other node. valsError[a][b] means the error of the ath node versus the bth connection
        if self.connection.type != "dense":
            self.biasWeights = np.random.rand(self.connection.nodes)*2 - 1 
            self.bias = np.random.random() 
            self.biasError = np.random.rand(self.connection.nodes)*2-1
            self.biasWeightsError = np.random.rand(self.connection.nodes)*2-1
        self.functionOut = np.random.rand(self.nodes)
    
    def forward(self): #compute the next values 
        self.connection.noActVals = self.weights.dot(self.vals) #dot product of weights and values
        if self.connection.type != "output":
            self.connection.vals = relu(self.connection.noActVals) #activating
            for i in range(len(self.connection.vals)): #add biases
                self.connection.vals[i] += self.bias*self.biasWeights[i] 
        else:
            self.connection.vals = sigmoid(self.connection.noActVals)

    def backward(self ):

        for i in range(len(self.vals)):
            for x in range(len(self.connection.vals)): #iterate through each weight 
                self.weightError[x][i] = self.vals[i] #weight error = (g'(x) * f'(g(x)) g(x) = weight*node so g'(x) = node. 
                if self.connection.type != "output":
                    self.weightError[x][i] *= step(self.connection.vals[x]) #f(x) = relu so f'(x) = step, so multiplying by step(g(x))
                else:
                    self.weightError[x][i] *= sigmoidDeriv(self.connection.vals[x]) #f(x) = relu so f'(x) = step, so multiplying by step(g(x))
        
        for i in range(len(self.vals)): #iterate through the node errors 
            for x in range(len(self.connection.vals)):
                if self.connection.type != "output":
                    self.valsError[x][i] = self.weights[x][i]*step(self.connection.vals[x]) # g(x) = node*weight so g'(x) = weight. step of the connection  valsError[x] corresponds to output[x]
                else:
                    self.valsError[x][i] = self.weights[x][i]*sigmoidDeriv(self.connection.vals[x]) # g(x) = node*weight so g'(x) = weight. step of the connection  valsError[x] corresponds to output[x]
                
        if self.connection.type != "output":
            for i in range(len(self.biasError)):
                self.biasError[i] = self.biasWeights[i]*step(self.connection.vals[i])

            for i in range(len(self.biasWeightsError)):
                self.biasWeightsError[i] = self.bias*step(self.connection.vals[i])




    def func2(self, targets, x, count): #for node errors
        
        if self.type == "output": #derivitive for output layer 
            arr = [] 
            for i in range(len(targets)):
                arr.append(2*(targets[i] - x[i])) #derivative of (t-x)^2 = 2(t-x) 
            
            return(arr)
        
        else:   #node derivative 
            arr = []          
            functionOut = np.array(self.connection.func2(targets, self.connection.vals, 0)) #variable to store the result of function
            error = np.array(self.valsError) #make into a np array
            error = error.T #transpose the array so it is formatted correctly 


            arr.append(error.dot(functionOut)) #how much does the node effect each of the next layers   
            
            
            for g in range(len(arr)):
                arr[g] = arr[g]/len(functionOut)
            
            #print("should be 1 num")
            
            arr = np.array(arr)
            self.functionOut = arr.squeeze(0)
            #print("sqeeze")
            #print(arr.squeeze(0))
            return(arr.squeeze(0))        
    def compedFunc2(self):
        return self.functionOut
    
    def func3(self, targets): #backpropogation algo
        
        arr = [] #instantiate array (size = self.weights)
        # SHOULD BE DONE
        if self.type == "input":
            functionOut = np.expand_dims(np.array(self.func2(targets, self.connection.vals, 0 )),1)
        else:
            functionOut = np.expand_dims(self.compedFunc2(),1)
        
        for i in range(len(self.weights)): #for each val error 
            error = np.array(self.weightError)
            arr2 = []
            for g in range(len(self.weightError[0])):
                arr2.append(error[i][g]*(functionOut[g][0])) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) 9self.weighterror * f'(vals)
            arr.append(arr2)
        return(arr)

        ## w1 corresponds to l1, need to average deriv for l1 (l1- o1 and l1 - o2)
        ## add np.dot() to add up all the derivitives and then divide ? 
    def func3Bias(self,targets):
        arr = [] 
        if self.type == "input":
            functionOut = np.expand_dims(np.array(self.connection.func2(targets, self.connection.vals, 0 )),1)
        else:
            functionOut = np.expand_dims(self.connection.compedFunc2(),1)
        arr2 = []
        error = np.array(self.biasWeightsError)

        for i in range(len(self.biasWeightsError)):
            arr2.append(error[i]*functionOut[i][0])
        arr.append(arr2)
        arr3 = 0
        error = np.array(self.biasError)
        for i in range(len(self.biasError)):
            arr3+=(error[i]*functionOut[i][0])
        arr3 = arr3/len(self.biasError)
        arr.append([arr3])
        return arr


    def update(self, targets, lr, loop): #start of backpropogation
        arr = self.func3(targets) #get errors of weights
        self.updateArr1.append(arr)
        if self.connection.type != "output":
            arr2 = self.func3Bias(targets)
            arr3 = arr2[0]
            arr4 = arr2[1]
            self.updateArr2.append(arr2)
            self.updateArr3.append(arr3)
            self.updateArr4.append(arr4)

        
        if loop % 40 == 0:
            for x in range(len(self.updateArr1)):

                

                for i in range(len(self.updateArr1[x])):
                    for p in range(len(self.updateArr1[x][0])):
                        self.weights[i][p] += lr*self.updateArr1[x][i][p] #update weights by error*lr 
                if self.connection.type != "output":
                    for i in range(len(self.updateArr3[x])):
                        self.biasWeights[i] += lr*self.updateArr3[x][i]*0.1
                    #print("arr4 ", arr4)
                        self.bias += lr*self.updateArr4[x][0]
            self.updateArr1 = [] 
            self.updateArr2 = [] 
            self.updateArr3 = [] 
            self.updateArr4 = []
        """
        old:
        arr = self.func3(targets) #get errors of weights
        if self.connection.type != "output":
            arr2 = self.func3Bias(targets)
            arr3 = arr2[0]
            arr4 = arr2[1]
        



        for i in range(len(arr)):
            for p in range(len(arr[0])):
                self.weights[i][p] += lr*self.arr[i][p] #update weights by error*lr 
        if self.connection.type != "output":
            for i in range(len(self.arr3)):
                self.biasWeights[i] += lr*self.arr3[i]
            #print("arr4 ", arr4)
            self.bias += lr*self.arr4[0]
    
    
        
        """

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
dense1 = dense(10)
dense2 = dense(6)
dense3 = dense(10)
out = output(2)
print("set")
input.type = "input"

input.connect(dense1)
dense1.connect(dense2)
dense2.connect(dense3)
dense3.connect(out)
print("set")
ins = np.random.rand(10000 ,2)
outs = []
#p = np.array([3,-2,5,5,0,-3,-3])#
#print(sigmoid(p))
#print(step(p))
for i in range(len(ins)):
    ins[i][0] *= 10
    ins[i][1] *= 10
c1 = 0 
c2 = 0
"""
for i in ins:

    if i[1] > (i[0]*2.2-0.5)**2:
        outs.append([0,1])
        c1 += 1
    else:
        outs.append([1,0])
        c2 += 1
"""
plot1 = []
plot2 = []
for i in ins:
    if i[1] > i[0]:
        outs.append([0,1])
        c1 += 1
        plot1.append(i)
    else:
        outs.append([1,0])
        c2 += 1
        plot2.append(i)
#plt.scatter(plot1[:][0], plot1[:][1])
#plt.scatter(plot2[:][0], plot2[:][1])
#plt.show()


print(c1,c2)
mses = [0,0]    
lr = 0.0003
epochs = 1000
aes = [0,0]
lastChange = 0
lastMses = [1,1]
for x in range(epochs):
    reds = []
    blues = []      
    for i in range(len(ins)):

        #input.input([1,2,2])
        input.input(ins[i])
        input.forward()
        dense1.forward()
        dense2.forward()
        dense3.forward()
        input.backward()
        dense1.backward()
        dense2.backward()
        dense3.backward()


        input.update(outs[i], lr, i)
        dense1.update(outs[i], lr, i)
        dense2.update(outs[i], lr, i)
        dense3.update(outs[i], lr, i)
        # input.input(ins[i])
        # input.forward()
        # dense1.forward()
        # dense2.forward()

        mses[0] += float(out.forward(outs[i])[0][0])
        mses[1] += float(out.forward(outs[i])[0][1])
        #aes[0] += float(out.forward(outs[i])[1][0])
        
        #aes[1] += float(out.forward(outs[i])[1][1])
        out1 = float(out.forward(outs[i])[1][1])
        out2 = float(out.forward(outs[i])[1][0])
        #print(out.forward(outs[i]))
        if out1 > out2:
            reds.append(ins[i])
            pass
        else: 
            blues.append(ins[i])
            #print(100000)
            #print(out1-out2)       
    
    print("epochs: ",x, "      mse = ", mses[0]/(len(ins)*(x+1)), mses[1]/(len(ins)*(x+1)))
    
    if lastMses[0] - mses[0]/(len(ins)*(x+1)) < 0 and lastMses[1] -  mses[1]/(len(ins)*(x+1)) <0 :
        if lastChange > 3:
            lr = lr/3
            print("lr = ", lr)
        lastChange = -1

    lastMses[0] =  mses[0]/(len(ins)*(x+1)) 
    lastMses[1] =  mses[1]/(len(ins)*(x+1)) 
    lastChange += 1
    if x % 5 == 0:
        reds = np.array(reds)
        blues = np.array(blues)
        try:
            plt.scatter(reds[:,0],reds[:,1])
            plt.scatter(blues[:,0],blues[:,1])
            plt.show()
            plt.clf()
        except:
            print("fail")
            if x == 0:
                raise 
            pass
    if x == 100:
        lr = lr/3 
    reds = []
    blues = []
mses[0] = mses[0]/(len(ins)*epochs)
mses[1] = mses[1]/(len(ins)*epochs)

print(mses)
print("Ae =", aes[0]/len(ins)*epochs, aes[1]/len(ins)*epochs)

ins = np.random.rand(300,2)
outs = []

for i in range(len(ins)):
    ins[i][0] *= 1
    ins[i][1] *= 1
"""
for i in ins:

    if i[1] > (i[0]*2.2-0.5)**2:
        outs.append([0,1])
        c1 += 1
    else:
        outs.append([1,0])
        c2 += 1
"""
for i in ins:
    if i[1] > i[0]:
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
print("weights : ", dense1.weights)
print("weights : ", dense2.weights)
print("weights : ", input.weights)
reds = np.array(reds)
blues = np.array(blues)
plt.scatter(reds[:,0],reds[:,1])
plt.scatter(blues[:,0],blues[:,1])
plt.show()

# y(x) = f(g(x)), g(x) = w*n + wb*b, f(x) = relu(x)
#y'(x) = g'(x)*f'(g(x)) = wb*step(out)
