# -*- coding: utf-8 -*-
import BackPropogation1 as bp1
import ForwardPropogation1 as fp1
import BackPropogation2 as bp2
import ForwardPropogation2 as fp2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#function to get data
def get_data(file_name):
    data = []
    label = []
    with open(file_name,"r") as fd:
        file = fd.readlines()
        for line in file:
            input = [i for i in line.strip("\n").split(",")]
            data.append([float(i) for i in input[:-1]])
            label.append(int(input[-1]))
    return data,label  
#function used for prediction
def predict(Batch_Norm, hidden_layer, model, data):
    if hidden_layer == 1:
       prob_scores, non_relevant = fp1.forward_prop(model, data)
    elif hidden_layer == 2:
       prob_scores, non_relevant1, non_relevant2 = fp2.forward_prop(Batch_Norm, model, data) 
    prediction = np.argmax(prob_scores, axis = 1)
    return prediction
#function to calculate error
def error(prediction, ground_truth) :
    count = 0
    n = len(ground_truth) 
    for i in range(n):
        if prediction[i] == ground_truth[i]:    
           count = count+1
    accuracy = float(count)/n
    return (1 - accuracy)*100
#function to calculate cross entropy error
def average_cross_entropy_error(Batch_Norm, hidden_layer, model, data, true_label):
    log_prob = []
    if hidden_layer == 1:
       prob_scores, non_relevant1 = fp1.forward_prop(model, data)
    elif hidden_layer == 2:
       prob_scores, non_relevant1, non_relevant2 = fp2.forward_prop(Batch_Norm, model, data)
    # loop through the rows  
    for i in range(len(data)):
       log_prob.append(-1*np.log(prob_scores[i, true_label[i]]))
    average_error = float(np.sum(log_prob))/len(data)
    return average_error
#function used for mean classification error
def mean_classification_error(Batch_Norm, hidden_layer, model, data, true_label):
    if hidden_layer == 1:
       prob_scores, non_relevant = fp1.forward_prop(model, data)
    elif hidden_layer == 2:
       prob_scores, non_relevant1, non_relevant2 = fp2.forward_prop(Batch_Norm, model, data) 
    #calculate the prediction  
    prediction = np.argmax(prob_scores, axis = 1)
    count = 0
    l = len(prediction)
    for i in range(l):
       if prediction[i] != true_label[i]:    
           count = count+1
    mean_error = (float(count)/l)*100
    return mean_error
 #function to initialize the parameters       
def initialize_parameters(hidden_layer, input_features, output_labels, hidden_units):
    # checks wheteher initialization is for 1 layer or two layer network
    if hidden_layer == 1:
       param = {}
       param['b1'] = np.zeros((1,hidden_units))
       param['b2'] = np.zeros((1,output_labels))
       np.random.seed(0)
       a1 =np.sqrt(6)/float(np.sqrt(100+784))
       a2 =np.sqrt(6)/float(np.sqrt(10+100))
       param['w1'] = np.random.uniform(-a1,a1,size = (input_features, hidden_units))
       param['w2'] = np.random.uniform(-a2,a2,size = (hidden_units, output_labels))
       #initialization for momentum update
       param['vb1'] = np.zeros((1,hidden_units))
       param['vb2'] = np.zeros((1,output_labels))
       param['vw1'] = np.zeros((input_features, hidden_units))
       param['vw2'] = np.zeros((hidden_units, output_labels))  
 
    elif hidden_layer == 2:  
       param = {}
       param['b1'] = np.zeros((1,hidden_units))
       param['b2'] = np.zeros((1,hidden_units))
       param['b3'] = np.zeros((1,output_labels))
       np.random.seed(0)
       a1 =np.sqrt(6)/float(np.sqrt(100+784))
       a2 =np.sqrt(6)/float(np.sqrt(100+100))
       a3 =np.sqrt(6)/float(np.sqrt(10+100))
       param['w1'] = np.random.uniform(-a1,a1,size = (input_features, hidden_units))
       param['w2'] = np.random.uniform(-a2,a2,size = (hidden_units, hidden_units))
       param['w3'] = np.random.uniform(-a3,a3,size = (hidden_units, output_labels))
       #initialization for momentum update
       param['vb1'] = np.zeros((1,hidden_units))
       param['vb2'] = np.zeros((1,hidden_units))
       param['vb3'] = np.zeros((1,output_labels))
       param['vw1'] = np.zeros((input_features, hidden_units))
       param['vw2'] = np.zeros((hidden_units, hidden_units))
       param['vw3'] = np.zeros((hidden_units, output_labels))
       #initialization for batch norm
       param['gamma1'] = np.ones((1,hidden_units))
       param['beta1'] = np.zeros((1,hidden_units))
       param['gamma2'] = np.ones((1,hidden_units))
       param['beta2'] = np.zeros((1,hidden_units))
       param['gamma3'] = np.ones((1,output_labels))
       param['beta3'] = np.zeros((1,output_labels))
    return param

def train_network(Batch_Norm, hidden_layer, X_train, Y_train, param, learning_rate, mini_batch, momentum, lamda):
    #implement mini batch gradient descent 
    if Batch_Norm == False:
       X_train, Y_train = shuffle(X_train,Y_train)       
       for i in range(0, X_train.shape[0], mini_batch):
           x = X_train[i:(i + mini_batch)]
           y = Y_train[i:(i + mini_batch)]
           #checks whether to run a 1 hidden layer network or 2 hidden layer network
           if hidden_layer == 1:   
              # forward propogation
              prob_scores,h1 = fp1.forward_prop(param,x)
              # backward propogation
              param = bp1.back_prop(prob_scores, param, learning_rate, x, y, h1, momentum, lamda) 
           elif hidden_layer == 2:
              # forward propogation
              prob_scores,h1,h2 = fp2.forward_prop(Batch_Norm,param,x)
              # backward propogation
              param = bp2.back_prop(Batch_Norm,prob_scores, param, learning_rate, x, y, h1, h2, momentum, lamda)       
       return param
     #if Batch_Norm is True implement a 2 hidden layer network
    else:
       X_train, Y_train = shuffle(X_train,Y_train) 
       #remove 24 traing samples to make it divisible by 32
       X_train = X_train[0:2976]
       Y_train = Y_train[0:2976]
       for i in range(0, X_train.shape[0], mini_batch):
           x = X_train[i:(i + mini_batch)]
           y = Y_train[i:(i + mini_batch)]
           # forward propogation
           prob_scores,h1,h2 = fp2.forward_prop(Batch_Norm,param,x)
           # backward propogation
           param = bp2.back_prop(Batch_Norm, prob_scores, param, learning_rate, x, y, h1, h2, momentum, lamda)       
       return param
         
         
         
        
if __name__ == "__main__":
    train_data, train_label = get_data("digitstrain.txt")
    valid_data, valid_label = get_data("digitsvalid.txt")
    test_data, test_label = get_data("digitstest.txt")
     
    #initialization block (10 lines)
    hidden_layer = 1
    Batch_Norm = False
    input_features = 784
    output_labels = 10
    hidden_units  = 100
    param = initialize_parameters(hidden_layer,input_features,output_labels,hidden_units)
    learning_rate = 0.1  #learning rate coefficiient
    momentum = 0.0       #momentum coefficient
    lamda = 0.0             #regularisation coefficient
    mini_batch = 30
    epoch_number = 400
    
    #implement batch norm block (4 lines)
#    Batch_Norm = True
#    hidden_layer = 2
#    param = initialize_parameters(hidden_layer,input_features,output_labels,hidden_units)
#    mini_batch = 32
    
    X_train = np.array(train_data)
    Y_train = np.array(train_label)
    
    train_error_list = []
    valid_error_list = []
    test_error_list = []
    train_cross_entropy_error_list = []
    valid_cross_entropy_error_list= []
    test_cross_entropy_error_list= []
    train_mean_classification_error_list = []
    valid_mean_classification_error_list= []
    
    for epoch in range(epoch_number):
        print('epoch:',epoch)
        # train the neural network with given number of epochs and mini_batch size
        param = train_network(Batch_Norm, hidden_layer, X_train,Y_train,param,learning_rate,mini_batch,momentum,lamda);
        # use the model for prediction
        train_cross_entropy_error = average_cross_entropy_error(Batch_Norm,hidden_layer,param,X_train,Y_train);
        train_mean_classification_error = mean_classification_error(Batch_Norm,hidden_layer,param,X_train,Y_train);
        train_prediction = predict(Batch_Norm,hidden_layer,param, X_train)
        train_error = error(train_prediction, Y_train)     
        valid_cross_entropy_error = average_cross_entropy_error(Batch_Norm,hidden_layer,param,np.array(valid_data),valid_label);
        valid_mean_classification_error = mean_classification_error(Batch_Norm,hidden_layer,param,np.array(valid_data),valid_label);
        valid_prediction = predict(Batch_Norm,hidden_layer,param, np.array(valid_data))
        valid_error = error(valid_prediction, valid_label)
        test_cross_entropy_error = average_cross_entropy_error(Batch_Norm,hidden_layer,param,np.array(test_data),test_label);
        test_prediction = predict(Batch_Norm,hidden_layer,param, np.array(test_data))
        test_error = error(test_prediction, test_label)
        #create the lists for plotting
        train_error_list.append(train_error)
        train_cross_entropy_error_list.append(train_cross_entropy_error)
        valid_error_list.append(valid_error)
        valid_cross_entropy_error_list.append(valid_cross_entropy_error)
        train_mean_classification_error_list.append(train_mean_classification_error)
        valid_mean_classification_error_list.append(valid_mean_classification_error)
        test_error_list.append(test_error)
        test_cross_entropy_error_list.append(test_cross_entropy_error)
    
    epochs = [x+1 for x in range(epoch_number)]    
    fig = plt.figure(1)
    plt.plot(epochs,train_error_list , 'b',label = 'train')   
    plt.plot(epochs,valid_error_list , 'g',label = 'validation')
    plt.plot(epochs,test_error_list , 'r',label = 'test')                     
    plt.title('General_Error')
    plt.legend()
    plt.show()
    fig.savefig("general_error.png")
    
    fig = plt.figure(2)
    plt.plot(epochs,train_cross_entropy_error_list , 'b',label = 'train')   
    plt.plot(epochs,valid_cross_entropy_error_list , 'g',label = 'validation')
    plt.plot(epochs,test_cross_entropy_error_list , 'r',label = 'test')
    plt.title('Cross_Entropy_Error')
    plt.legend()
    plt.show()
    fig.savefig("cross_entropy_error.png")
    
#    fig = plt.figure(3)
#    plt.plot(epochs,train_mean_classification_error_list , 'b',label = 'train')   
#    plt.plot(epochs,valid_mean_classification_error_list , 'g',label = 'validation')
#    plt.title('mean_classification_Error')
#    plt.legend()
#    plt.show()
#    fig.savefig("mean_classification_error.png")
    
    
    
    
    #plot w
    w_reshaped = []
    W = param['w1']  
    fig, axes = plt.subplots(10, 10)
    vmin, vmax = W.min(), W.max()
    for coef, ax in zip(W.T, axes.ravel()):
        ax.imshow(coef.reshape(28, 28), cmap = 'gray')
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()
    fig.savefig("visualisation_W.png")
   

    
    