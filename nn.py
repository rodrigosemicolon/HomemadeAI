import numpy as np

np.random.seed(seed=2)

def linear(input):
    return input

def relu(input):
    return [max(i,0) for i in input]

def sigmoid(input):
    return [1/(1*np.e**-i) for i in input]

def softmax(input):
    return np.exp(input)/np.sum(np.exp(input), axis=0)

def MeanSquareLoss(results, benchmarks):
    n = len(results)
    res = benchmarks-results
    res=res**2
    return res/n
    
class layer():
    def __init__(self, inputs, size, activation_function, weights=None, biases=None):
        
        self.size=size

        if(weights==None):
            self.weights = np.random.rand(inputs,size)
        else:
            assert len(weights)==size
            self.weights=weights

        if(biases==None):
            self.biases = np.random.rand(size)
        else:
            assert len(biases)==size
            self.biases=biases
        
        self.activation_function=activation_function

    def forward(self, x):
        return self.activation_function(np.dot(x,self.weights) + self.biases)

    def __str__(self):
        return f"size:{self.size}\nweights:{self.weights}\nbiases:{self.biases}\n"

    def __repr__(self):
        return f"size:{self.size}\nweights:{self.weights}\nbiases:{self.biases}\n"
    

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
        
    def forward(self, x):
        
        for layer in self.layers:
            x=layer.forward(x)
            print("output: ",x)
        return x
    
    def train(self, X, y, batch_size, learning_rate):
        pass


    def test(self, X, y):
        pass



t1=layer(3,4,linear)
t2=layer(4,2,linear)

print("t1:n",t1,"\n")
print("t2:n",t2,"\n")

test=nn([(2,linear),(3,linear), (2, softmax)])
test.forward([1,0])
