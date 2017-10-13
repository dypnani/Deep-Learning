# -*- coding: utf-8 -*-
import numpy as np
import helper 
import BatchNorm

def forward_prop(Batch_Norm, param, x):
    w1,w2,w3,b1,b2,b3 = param['w1'],param['w2'],param['w3'],param['b1'],param['b2'],param['b3']
    # input to hidden layer- pre activation
    a1 = np.dot(x,w1) + b1
    
    if Batch_Norm == True:
      #send it to batch norm
      a1,param = BatchNorm.forward(a1, param, level = 1)
    
    #hidden layer activation
    h1 = helper.sigmoid(a1)
    #hidden layer to hidden layer - pre-activation
    a2 = np.dot(h1,w2) + b2
     
    if Batch_Norm == True:
      #send it to batch norm
      a2,param = BatchNorm.forward(a2, param, level = 2)
    
    #hidden layer activation
    h2 = helper.sigmoid(a2) 
    #hidden layer to output - pre-activation
    a3 = np.dot(h2,w3) + b3
    
    if Batch_Norm == True:
      #send it to batch norm
      a3,param = BatchNorm.forward(a3, param, level = 3)
      
    #output layer activation resulting in probability scores
    prob_scores = helper.softmax(a3)
    return prob_scores,h1,h2
