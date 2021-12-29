import numpy as np

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
        self.outputs=np.zeros(size)      
        self.activation_function=activation_function
    
    def update_weights(self,weights):
        assert len(weights)==len(self.weights)
        self.weights=weights

    def forward(self, x):
        self.outputs=self.activation_function.activation(np.dot(x,self.weights) + self.biases)
        return self.outputs

    def last_outputs(self):
        return self.outputs
    def __str__(self):
        return f"size:{self.size}\nweights:{self.weights}\nbiases:{self.biases}\n"

    def __repr__(self):
        return f"size:{self.size}\nweights:{self.weights}\nbiases:{self.biases}\n" 

