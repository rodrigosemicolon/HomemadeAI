import numpy as np

class linear():
    def activation(self,input):
        return input

    def derivative(self,input):
        return 1

class relu():

    def activation(self,input):
        return [max(i,0) for i in input]
    
    def derivative(self, input):
        #considering point 0 as having derivative 0 though it is undefined
        if input>0: 
            return 1
        else:
            return 0


class sigmoid():

    def activation(self,input):
        return [1/(1*np.e**-i) for i in input]

    def derivative(self,input):
        sig = self.activation(input)
        return sig*(1-sig)

class softmax():

    def activation(self,input):
        return np.exp(input)/np.sum(np.exp(input), axis=0)

    def derivative(self,input):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = input.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)