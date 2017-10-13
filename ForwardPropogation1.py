# -*- coding: utf-8 -*-
import numpy as np
import helper 
def forward_prop(param, x):
    w1,w2,b1,b2 = param['w1'],param['w2'],param['b1'],param['b2']
    # input to hidden layer- pre activation
    a1 = np.dot(x,w1) + b1
    #hidden layer activation
    h1 = helper.sigmoid(a1)
    #h1 = helper.tanh(a1) #for tanh uncomment this
    #h1 = helper.relu_activation(a1)
    #input to output layer - pre-activation
    a2 = np.dot(h1,w2) + b2
    #output layer activation resulting in probability scores
    prob_scores = helper.softmax(a2)
    return prob_scores,h1