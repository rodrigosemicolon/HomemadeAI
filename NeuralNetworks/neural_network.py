import numpy as np
from activation_functions import linear, relu, sigmoid, softmax
from layer import layer


np.random.seed(seed=2) #for debugging



def MeanSquareLoss(results, benchmarks):
    n = len(results)
    res = benchmarks-results
    res=res**2
    res=np.sum(res)
    return res/n
    

class nn():
    #layers as a list of tuples containing number of nodes and activation function
    def __init__(self, layers, custom_layers=[]):
        if len(custom_layers)>0:
            assert len(custom_layers)>2
            self.layers=custom_layers
        else:
            assert len(layers)>2
            
            self.layers=[]
            for l in range(len(layers)):
                if l==0:
                    current_layer = layer(layers[l][0],layers[l][0],layers[l][1])
                else:
                    current_layer = layer(layers[l-1][0],layers[l][0],layers[l][1])    
                self.layers.append(current_layer)
        self.nLayers = len(self.layers)

    def forward(self, x):
        
        for layer in self.layers:
            x=layer.forward(x)
            print("output: ",x)
        return x
    def backpropagate(self,outputs, expected):
        delta={}
        for i in reversed(range(self.nLayers)):
            if i==len(self.nLayers)-1:
                delta[i] = (outputs-expected)*[self.activation.derivative(i) for i in outputs]
            else:
                error=0.0
                #errors[i] = 
    def train(self, X, y, batch_size, learning_rate):
        for i in range(0,len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            loss = []
            errors=[]
            for j,k in enumerate(batch_X):
                output = self.forward(k)
                loss.append(MeanSquareLoss(output,batch_y[j]))
                errors.append(output-batch_y[j])
            loss = np.sum(loss)#

    def test(self, X, y):
        pass


"""
t1=layer(3,4,sigmoid())
t2=layer(4,2,sigmoid())
"""
#print("t1:n",t1,"\n")
#print("t2:n",t2,"\n")

#print(MeanSquareLoss(np.array([0.8,0.1,0.1]),np.array([1,0,0])))