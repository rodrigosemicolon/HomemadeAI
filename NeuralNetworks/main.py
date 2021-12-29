from neural_network import nn
from activation_functions import *

sig = sigmoid()
sof = softmax()
test=nn([(2,sig),(3,sig), (2, sof)])
test.forward([1,0])