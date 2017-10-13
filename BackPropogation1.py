# -*- coding: utf-8 -*-
import numpy as np
import helper

def back_prop(prob_scores, param, lRate, x, y, h1, momentum, lamda): 
    w1,w2,b1,b2 = param['w1'],param['w2'],param['b1'],param['b2']
    vw1,vw2,vb1,vb2 = param['vw1'],param['vw2'],param['vb1'],param['vb2']
    
    #gradient wrt output layer preactivation
    da2 = prob_scores.copy()
    for i in range(len(x)):
        da2[i,y[i]] -= 1
    da2 /= len(x)
    dw2 = np.dot(h1.T, da2)
    db2 = np.sum(da2, axis = 0, keepdims = True) 
    da1 = np.dot(da2, w2.T)*(helper.d_sigmoid(h1))   
  #  da1 = np.dot(da2, w2.T)*(helper.d_tanh(h1))  #for tanh activation uncomment this
  #  da1 = np.dot(da2, w2.T)   #for RELU activation uncomment this
  #  da1[h1 <= 0] = 0          ##for RELU activation uncomment this    
    dw1 = np.dot(x.T, da1)
    db1 = np.sum(da1, axis = 0, keepdims = True)
    
    # adding regularisation term
    dw1 += lamda*w1
    dw2 += lamda*w2
    # momentm update
    vw1 = (momentum*vw1) - (lRate*dw1)
    vw2 = (momentum*vw2) - (lRate*dw2)
    vb1 = (momentum*vb1) - (lRate*db1)
    vb2 = (momentum*vb2) - (lRate*db2)
    #update the parameters
    w1 += vw1
    w2 += vw2
    b1 += vb1
    b2 += vb2

    param['w1'],param['w2'],param['b1'],param['b2'] = w1,w2,b1,b2
    param['vw1'],param['vw2'],param['vb1'],param['vb2'] = vw1,vw2,vb1,vb2
    return param