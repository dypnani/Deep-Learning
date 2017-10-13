# Deep-Learning
The code for this neural network implementation consists of :

main file for execution: 
NeuralNetwork.py --- this is the file for which we make changes in main function and execute

support files: (these are imported whereever necessary)
ForwardPropogation1.py ----implements code for 1 hidden layer neural network forward propogation
ForwardPropogation2.py --- implements code for 2 hidden layer neural network forward propogation
BackwardPropogation1.py -- implements code for 1 hidden layer neural network Backward propogation
BackwardPropogation2.py -- implements code for 2 hidden layer neural network Backward propogation
helper.py -----------------implements helper functions
BatchNorm.py -----------------implements the batch normalisation on 2 layer network

Instructions:

Make sure all the files given above are in same folder
No of hidden layers used is given by variable hidden_layer. if hidden_layer = 1 means one hidden layer and if hidden_layer = 2 means two hidden layers used.
Any changes required, can be made in #Initialization block of main function in NeuralNetwork.py for variables momentum, hidden unit number, lamda for regularisation  


uncomment the necessary functions being used in forward and backward propogation  
To implement batch normalization uncomment the #implement batch norm block (4 lines) in main function in NeuralNet.py 

Thank You
