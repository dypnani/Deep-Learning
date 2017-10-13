# -*- coding: utf-8 -*-
import numpy as np
import helper
import BatchNorm

def back_prop(Batch_Norm, prob_scores, param, lRate, x, y, h1, h2, momentum, lamda): 
    w1,w2,w3,b1,b2,b3 = param['w1'],param['w2'],param['w3'],param['b1'],param['b2'],param['b3']
    vw1,vw2,vw3,vb1,vb2,vb3 = param['vw1'],param['vw2'],param['vw3'],param['vb1'],param['vb2'],param['vb3']
    gamma1,beta1,gamma2,beta2,gamma3,beta3 = param['gamma1'],param['beta1'],param['gamma2'],param['beta2'],param['gamma3'],param['beta3']
    #gradient wrt output layer preactivation
    da3 = prob_scores.copy()
    for i in range(len(x)):
        da3[i,y[i]] -= 1
    da3 /= len(x)
    #back prop for batch normalization
    if Batch_Norm == True:
       da3, dgamma3, dbeta3 = BatchNorm.backprop(da3,param, level = 3) 
       gamma3 -= (lRate*dgamma3)
       beta3 -= (lRate*dbeta3)
       param['gamma3'], param['beta3'] = gamma3,beta3
    
    dw3 = np.dot(h2.T, da3)
    db3 = np.sum(da3, axis = 0, keepdims = True) 
    da2 = np.dot(da3, w3.T)*(helper.d_sigmoid(h2))
    
    #back prop for batch normalization
    if Batch_Norm == True:
       da2, dgamma2, dbeta2 = BatchNorm.backprop(da2,param,level = 2) 
       gamma2 -= (lRate*dgamma2)
       beta2 -= (lRate*dbeta2)
       param['gamma2'], param['beta2'] = gamma2,beta2
    
    dw2 = np.dot(h1.T, da2)
    db2 = np.sum(da2, axis = 0, keepdims = True)
    da1 = np.dot(da2, w2.T)*(helper.d_sigmoid(h1))
    
    #back prop for batch normalization
    if Batch_Norm == True:
       da1, dgamma1, dbeta1 = BatchNorm.backprop(da1,param,level = 1)
       gamma1 -= (lRate*dgamma1)
       beta1 -= (lRate*dbeta1)
       param['gamma1'], param['beta1'] = gamma1,beta1
    
    dw1 = np.dot(x.T, da1)
    db1 = np.sum(da1, axis = 0, keepdims = True)
    
    # adding regularisation term
    dw1 += lamda*w1
    dw2 += lamda*w2
    dw3 += lamda*w3
    
    # momentm update
    vw1 = (momentum*vw1) - (lRate*dw1)
    vw2 = (momentum*vw2) - (lRate*dw2)
    vw3 = (momentum*vw3) - (lRate*dw3)
    vb1 = (momentum*vb1) - (lRate*db1)
    vb2 = (momentum*vb2) - (lRate*db2)
    vb3 = (momentum*vb3) - (lRate*db3)

    
    w1 += vw1
    w2 += vw2
    w3 += vw3
    b1 += vb1
    b2 += vb2
    b3 += vb3

    
    param['w1'],param['w2'],param['w3'],param['b1'],param['b2'],param['b3'] = w1,w2,w3,b1,b2,b3
    param['vw1'],param['vw2'],param['vw3'],param['vb1'],param['vb2'],param['vb3']= vw1,vw2,vw3,vb1,vb2,vb3
    
    return param