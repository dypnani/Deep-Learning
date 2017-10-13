# -*- coding: utf-8 -*-

import numpy as np

def forward(x, param,level):
    if level == 1:
       gamma,beta = param['gamma1'], param['beta1']
    elif level == 2:
       gamma,beta = param['gamma2'], param['beta2'] 
    elif level == 3:
       gamma,beta = param['gamma3'], param['beta3'] 
       
    mean = np.mean(x, axis = 0, keepdims = True)
    var= np.var(x, axis = 0, keepdims = True)
    sd = np.sqrt(var + 1e-6)
    xnorm = (x- mean)/sd
    output = (gamma*xnorm) + beta  
    
    if level == 1:
       param['x1']= x
       param['xnorm1'] = xnorm
       param['mean1'] = mean
       param['var1'] = var
       
    elif level == 2:
       param['x2']= x
       param['xnorm2'] = xnorm
       param['mean2'] = mean
       param['var2'] = var
    elif level == 3:
       param['x3']= x
       param['xnorm3'] = xnorm
       param['mean3'] = mean
       param['var3'] = var
        
    return output,param

def backprop(dout, param, level):
    if level == 1:
       x = param['x1']
       xnorm = param['xnorm1']
       mean = param['mean1']
       var = param['var1']
       gamma = param['gamma1']
    elif level == 2:
       x = param['x2']
       xnorm = param['xnorm2'] 
       mean = param['mean2'] 
       var = param['var2'] 
       gamma = param['gamma2']
    elif level == 3:
       x = param['x3']
       xnorm = param['xnorm3'] 
       mean = param['mean3'] 
       var = param['var3'] 
       gamma = param['gamma3']
    
    N = len(x)
    sd = np.sqrt(var + 1e-6)
    dbeta = np.sum(dout, axis = 0, keepdims =True)
    dgamma = np.sum(dout*xnorm, axis = 0, keepdims = True)
    # used the derivation done for dx as shown in 6(h) answer in pdf document
    dx = (1.0 / N) * gamma * (1.0/sd) * (N * dout - dbeta - 
         (x-mean)*(1.0/np.power(sd,2))* np.sum(dout*(x-mean), axis=0, keepdims = True))
    
    return dx, dgamma, dbeta
    
    