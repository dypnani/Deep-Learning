# -*- coding: utf-8 -*-
import numpy as np

# helper functions for sigmoid and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)

def softmax(x):
    c = np.max(x, axis =1 ,keepdims =True )
    x = x-c
    exp = np.exp(x)
    return exp/np.sum(exp, axis = 1, keepdims =True )

def relu_activation(x):
    return np.maximum(x, 0)

def tanh(x):
    return (2*sigmoid(2*x) - 1)

def d_tanh(x):
    return (1 - np.power(x,2))
