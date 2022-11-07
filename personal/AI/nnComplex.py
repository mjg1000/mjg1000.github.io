import matplotlib.pyplot as plt
import numpy as np 
import math
def relu(x): #relu activation function 
    return np.maximum(0,x)      
def step(x): #relu derivative 
    if type(x) == list: #function works even for a list input 
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
def sigmoid(x): #final layer activation 
    #x = np.clip(x,-100,100)
    return(1/(1+2.72**(-x)))
def sigmoidDeriv(x): 
    sig = sigmoid(x)
    return (sig*(1-sig))
def cReluSquare(x):
    if x.real**2 - x.imag**2 > 0:
        return x 
    else:
        return 0 + 0j
def cReluPart(x):
    ans = 0 + 0j 
    if x.real > 0:
        ans += x.real 
    if x.imag > 0: 
        ans += x.imag 
    return ans 
def cReluSquareDeriv(x):
    if (x.real**2 > x.imag**2):
        return 1 + 1j 
    else: 
        return 0 
def cReluPartDeriv(x):
    ans = 0 + 0j 
    if x.real > 0: 
        ans += 1 
    if x.imag > 0:
        ans += 1j 
    return ans 



class dense(): #dense class for standard hidden layer 
    def __init__(self,nodes): #set up all default values

        self.nodes = nodes #number of nodes in the layer 
        self.noActVals = np.array(nodes) #the value of each node, without being activated
        self.vals = np.array(nodes) #the value of each node
        
        self.type = "hidden" #if this layer is an output 

        #used for backpropogation so batches and mini-batches work
        self.updateArr1 = [] 
        self.updateArr2 = []
        self.updateArr3 = []
        self.updateArr4 = []
    
    def input(self, values): #feed values into input layer  
        self.vals = values #set each value for the input layer 

    def connect(self, connection): #link up nodes and weights to the next layer 

        self.connection = connection #the next layer 
        
        self.weights = np.random.rand(self.connection.nodes,self.nodes)*2-1 +1.j*np.random.rand(self.connection.nodes,self.nodes)*2-1j #the weights. w[x][:] corresponds to output[x], w[:][x] corresponds to input[x]
        self.weightError = np.random.rand(self.connection.nodes,self.nodes)*2-1 +1.j*np.random.rand(self.connection.nodes,self.nodes)*2-1j #the error of each of the weights
        self.valsError = np.random.rand(self.connection.nodes,self.nodes)*2-1 +1.j*np.random.rand(self.connection.nodes,self.nodes)*2-1j #the error of each of the nodes with respect to each other node. valsError[a][b] means the error of the ath node versus the bth connection
        
        if self.connection.type != "dense": #add a bias provided that the next layer is not the output (output layer is just the cost function so a bias messes things up )
            self.biasWeights = np.random.rand(self.connection.nodes)*2-1 +1.j*np.random.rand(self.connection.nodes)*2-1j #weights and errors for the bias 
            self.bias = np.random.random() +1.j*np.random.random()
            self.biasError = np.random.rand(self.connection.nodes)*2-1+1.j*np.random.rand(self.connection.nodes)*2-1j #error of the bias (relative to each next node)
            self.biasWeightsError = np.random.rand(self.connection.nodes)*2-1+1.j*np.random.rand(self.connection.nodes)*2-1j#error of the bias weights (relative to corresponding nodes)

        self.functionOut = np.random.rand(self.nodes) +1.j*np.random.rand(self.nodes) #var to store derivatives when it has already been computed to speed up processing time 
    
    def forward(self): #compute the next values  
        self.connection.noActVals = self.weights.dot(self.vals) #dot product of weights and values without the activation - dot product will give the unactivated values for the next layer 
        if self.connection.type != "output": #output layer doesnt use a relu activation or a bias so needs special exception  
            self.connection.vals = relu(self.connection.noActVals) #passing the values through the activation function 
            for i in range(len(self.connection.vals)): #add biases to each node 
                self.connection.vals[i] += self.bias*self.biasWeights[i] 
        else:
            self.connection.vals = sigmoid(self.connection.noActVals)

    def backward(self ): #compute layer-wise derivatives aka node1 - next layers or weight1 to node1 

        for i in range(len(self.vals)): # weight errors 
            for x in range(len(self.connection.vals)): #iterate through each weight 
                self.weightError[x][i] = self.vals[i] #weight error = (g'(x) * f'(g(x)) g(x) = weight*node so g'(x) = node. 
                if self.connection.type != "output": #output uses sigmoid and so diff derivative 
                    self.weightError[x][i] *= step(self.connection.vals[x]) #f(x) = relu so f'(x) = step, so multiplying by step(g(x))
                else:
                    self.weightError[x][i] *= sigmoidDeriv(self.connection.vals[x]) #f(x) = relu so f'(x) = step, so multiplying by step(g(x))
        
        for i in range(len(self.vals)): #iterate through the node errors 
            for x in range(len(self.connection.vals)):
                if self.connection.type != "output":
                    self.valsError[x][i] = self.weights[x][i]*step(self.connection.vals[x]) # g(x) = node*weight so g'(x) = weight. step of the connection  valsError[x] corresponds to output[x]
                else:
                    self.valsError[x][i] = self.weights[x][i]*sigmoidDeriv(self.connection.vals[x]) # g(x) = node*weight so g'(x) = weight. step of the connection  valsError[x] corresponds to output[x]
                
        if self.connection.type != "output": #erros for the bias - output has no bias 
            for i in range(len(self.biasError)):
                self.biasError[i] = self.biasWeights[i]*step(self.connection.vals[i])

            for i in range(len(self.biasWeightsError)):
                self.biasWeightsError[i] = self.bias*step(self.connection.vals[i])




    def func2(self, targets, x, count): #for node errors (count var is useless ) tbh this function is a giant mess because derivatives are HARD, but it computes the derivative between nodes
        
        if self.type == "output": #derivitive for output layer 
            arr = [] 
            for i in range(len(targets)):
                arr.append(2*(targets[i] - x[i])) #derivative of (t-x)^2 = 2(t-x) 
            self.functionOut = arr
            return(arr)
        
        else:   #node derivative 
            arr = []          
            functionOut = np.array(self.connection.func2(targets, self.connection.vals, 0)) #variable to store the result of function, this is a recursive statement from the inpute ----> cost function (mean squared error)
            error = np.array(self.valsError) #make into a np array
            error = error.T #transpose the array so it is formatted correctly 


            arr.append(error.dot(functionOut)) #how much does the node effect each of the next layers   
            
            
            for g in range(len(arr)): #take the average, n01 affects n11,n12,n13 etc so take the average impact 
                arr[g] = arr[g]/len(functionOut)
            
            
            arr = np.array(arr)#remove excess dimension 
            arr = arr.squeeze(0)
            for i in range(len(arr)):
                #print(arr)
                #print(arr[i])
                if arr[i] > 3:
                    if np.random.randint(1,10000) == 30:
                        print("clip up")
                    
                    arr[i] = 3
                elif arr[i] < -3:
                    arr[i] = -3 
                    if np.random.randint(1,10000) == 30:
                        print("clip down")
                    #print("clip") 
            self.functionOut = arr #store result for other layers 
            return(arr)        
    def compedFunc2(self):
        return self.functionOut
    
    def func3(self, targets): #backpropogation algo for the weights and biases to the nodes in the next layer 
        
        arr = [] #instantiate array (size = self.weights)
        # SHOULD BE DONE
        if self.type == "input":
            functionOut = np.expand_dims(np.array(self.connection.func2(targets, self.connection.vals, 0 )),1)
        else: #the input layer should compute the derivatives for all other layers 
            functionOut = np.expand_dims(self.connection.compedFunc2(),1)
        #print("____________")
        #print(functionOut)
        for i in range(len(self.weights)): #for each val error 
            error = np.array(self.weightError)
            arr2 = []
            for g in range(len(self.weightError[0])):
                arr2.append(error[i][g]*(functionOut[i][0])) #self.vals[p].dot        g'(x) * f'(g(x)) for each weight, new weight should be g'(x) = self.weighterror * f'(vals)
            arr.append(arr2)
        return(arr)

        ## w1 corresponds to l1, need to average deriv for l1 (l1- o1 and l1 - o2)
        ## add np.dot() to add up all the derivitives and then divide ? 
    def func3Bias(self,targets): #fderivatives for the bias, essentially the exact same as func3 above 
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


    def update(self, targets, lr, loop, epoch): #start of backpropogation, actually updates all the values 
        arr = self.func3(targets) #get errors of weights
        self.updateArr1.append(arr) #smth like a 3d or 4d array to store alllll the updates for all the different inputs 
        if self.connection.type != "output": #output has no bias 
            arr2 = self.func3Bias(targets)
            arr3 = arr2[0]
            arr4 = arr2[1]
            self.updateArr2.append(arr2)
            self.updateArr3.append(arr3)
            self.updateArr4.append(arr4)

        
        if loop % 32 == 0 or (loop < 10000 and epoch == 1): #loop mod x, x = batch size
            for x in range(len(self.updateArr1)): #for each batch 

                

                for i in range(len(self.updateArr1[x])): #for each weight 
                    for p in range(len(self.updateArr1[x][0])):
                        self.weights[i][p] += lr*self.updateArr1[x][i][p] #update weights by error*lr - error is exactly the derivative 
                if self.connection.type != "output": #bias stuff 
                    for i in range(len(self.updateArr3[x])):
                        self.biasWeights[i] += lr*self.updateArr3[x][i]
                    #print("arr4 ", arr4)
                    self.bias += lr*self.updateArr4[x][0]
            self.updateArr1 = []  #get rid of the stored input errors
            self.updateArr2 = [] 
            self.updateArr3 = [] 
            self.updateArr4 = []
       
class output(dense): #output layer is slightly different 
    def __init__(self, nodes):
        super().__init__(nodes)
        self.type = "output"
    
    def forward(self, targets):
        arr = [] 
        for i in range(len(self.vals)):
            arr.append((targets[i]-self.vals[i])**2)
        return(arr,self.vals)

#NETWORK BEGINS:
#create layers 
network = [] 
network.append(dense(2))
network.append(dense(8))
network.append(dense(16))
network.append(dense(32))
network.append(dense(16))
network.append(dense(8))
#network.append(dense(2))
network.append(output(2))

print("set")
#show input is an input
network[0].type = "input"

#connect all the layers together    
for i in range(len(network)-1):
    network[i].connect(network[i+1])
print("set")
#randomly create a set of inputs for the network 
ins = np.random.rand(5000 ,2)
outs = []
for i in range(len(ins)): #expand the size of possible inputs to from 0 - 10 
    ins[i][0] *= 10
    ins[i][1] *= 10
c1 = 0 
c2 = 0
plot1 = [] #graph testing so I can graph the decision boundary 
plot2 = []

for i in ins:
    #if 10 > (i[0]-5)**2+2*(i[1]-5)**2:
    #if 3*math.sin(i[0]/1.5) +4 > i[1]:
    #if math.sinh(i[0])/math.sin(i[0]) + math.cosh(i[0])/math.cos(i[1]) + math.tanh(i[1])/math.tan(i[1]) < 3:
    #if i[1]> (i[0]):
    #if i[1] > i[0]**2:
    #if i[1] >i[0]*math.tan(math.sqrt(i[0]**2+i[1]**2)):
    if 5*math.sin(1/(0.01*(i[0]+3)))+7 > i[1]:
        outs.append([0,1])
        c1 += 1
        plot1.append(i)
    else:
        outs.append([1,0])
        c2 += 1
        plot2.append(i)
plot1 = np.array(plot1)
plot2 = np.array(plot2)
plt.scatter(plot1[:,0],plot1[:,1])
plt.scatter(plot2[:,0],plot2[:,1])
plt.show()
plt.ion()
plt.show()
lr = 0.01
epochs = 1000
def train(lr, epochs, network, ins, outs, loop):  
    mseGraph = []
    aes = [0,0]
    lastChange = 0
    lastMses = [1,1]
    print(c1,c2)    
    mses = [0,0]  
    for x in range(epochs): #loop for epochs 
        mses2 = [0,0]
        reds = [] #create the graph points 
        blues = []  
        redVals = []
        blueVals = []     
        for i in range(len(ins)): # loop through inputs 

            #TRAINING LOOP
            network[0].input(ins[i])
            for g in range(len(network)-1):
                #input input 
                network[g].forward()    
            for g in range(len(network)-1):
                network[g].backward() #get errors 
            for g in range(len(network)-2):
                network[g].update(outs[i], lr, i, x) #Update weights based on errors 
            #TRAINING LOOP END 

            #Mean squared errors (final layer returns raw values and mses):
            forwardAns = network[-1].forward(outs[i])
            msNow1 = float(forwardAns[0][0])
            msNow2 = float(forwardAns[0][1])
            
            mses[0] += msNow1
            mses[1] += msNow2
            mses2[0] += msNow1
            mses2[1] += msNow2
            #Raw values:
            out1 = float(forwardAns[1][0])
            out2 = float(forwardAns[1][1])
            redVals.append([ins[i][0],ins[i][1],out1])
            blueVals.append([ins[i][0],ins[i][1],out2])
            #PLOTTING DECISION BOUNDARY
            if out1 > out2:
                reds.append(ins[i]) #if it is class1, add to red, else add to blues 
                pass
            else: 
                blues.append(ins[i])
                
        
        print("epochs: ",x, "      mse = ", mses2[0]/(len(ins)), mses2[1]/(len(ins))) #give data 
        mseGraph.append([mses2[0],mses2[1],x])
        if lastMses[0] - mses[0]/(len(ins)*(x+1)) < 0 and lastMses[1] -  mses[1]/(len(ins)*(x+1)) <0 : #sometimes decrease lr if its losing progress 
            if lastChange > 1:  
                lr = lr/3
                print("lr = ", lr)
            lastChange = -1
        if x == 0:
            lr = lr/2
        if x == 20:
            lr = lr/2 
        
        lastMses[0] =  mses[0]/(len(ins)*(x+1)) 
        lastMses[1] =  mses[1]/(len(ins)*(x+1)) 
        lastChange += 1
        print(mseGraph)
        if x % 1 == 0: #scatterplot every fith epoch
            reds = np.array(reds)
            blues = np.array(blues)
            print(len(reds))
            print(len(blues))
            try:
                plt.clf()
                plt.scatter(reds[:,0],reds[:,1])
                plt.scatter(blues[:,0],blues[:,1])
                plt.draw()
                #plt.show()
                #plt.show(block=False)
                plt.pause(1)
                print("OK")
                #plt.clf()
                
                pass
            except:
                print("fail")
                if x == 0:
                    pass
                    #raise 
                pass
            #heatmap 
            """
            mapRed = np.zeros((101,101))
            mapBlue = np.zeros((101,101))
            for i in redVals:
                mapRed[round(i[0]*10)][round(i[1]*10)] = i[2]
            for i in blueVals:
                mapBlue[round(i[0]*10)][round(i[1]*10)] = i[2]
            fig,ax = plt.subplots()
            im = ax.imshow(mapRed)              
            fig.tight_layout()
            plt.show()      
            plt.clf()
            fig,ax = plt.subplots()
            im = ax.imshow(mapBlue)              
            fig.tight_layout()
            plt.show()
            plt.clf()
            """

        if x == 100:
            lr = lr/3 
        reds = []
        blues = []
    
    mseGraph = np.array(mseGraph)

    #axs[loop].plot(mseGraph[:,2],mseGraph[:,1])
    #axs[loop].plot(mseGraph[:,2],mseGraph[:,0])
    #reds = np.array(reds)
    #blues = np.array(blues)
    #print(reds)  
    """  
    try:
        
        axs[loop // 4 ][loop%5].scatter(reds[:,0],reds[:,1])
        axs[loop // 4][loop%5].scatter(blues[:,0],blues[:,1])
        
        pass
    except:
        print("fail")
        if x == 0:
            #pass
            raise 
        pass
    """
    #plt.show()
    #plt.clf()

    mses[0] = mses[0]/(len(ins)*epochs)
    mses[1] = mses[1]/(len(ins)*epochs)

    print(mses)
    print("Ae =", aes[0]/len(ins)*epochs, aes[1]/len(ins)*epochs)
    return network
network = train(lr, epochs,network, ins, outs,i )
plt.show()
#Validation stuff

