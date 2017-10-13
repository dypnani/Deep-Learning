# Deep-Learning
The code for this neural network implementation consists of :   <br />

Main file for execution:  <br />
NeuralNetwork.py --- this is the file for which we make changes in main function and execute <br />

Support files: (these are imported whereever necessary) <br />
ForwardPropogation1.py ----implements code for 1 hidden layer neural network forward propogation <br />
ForwardPropogation2.py --- implements code for 2 hidden layer neural network forward propogation <br />
BackwardPropogation1.py -- implements code for 1 hidden layer neural network Backward propogation <br />
BackwardPropogation2.py -- implements code for 2 hidden layer neural network Backward propogation <br />
helper.py -----------------implements helper functions <br />
BatchNorm.py -----------------implements the batch normalisation on 2 layer network <br />

Instructions: <br />

Make sure all the files given above are in same folder <br />
No of hidden layers used is given by variable hidden_layer. if hidden_layer = 1 means one hidden layer and if hidden_layer = 2 means two hidden layers used. <br />
Any changes required, can be made in #Initialization block of main function in NeuralNetwork.py for variables momentum, hidden unit number, lamda for regularisation   <br />


uncomment the necessary functions being used in forward and backward propogation   <br />
To implement batch normalization uncomment the #implement batch norm block (4 lines) in main function in NeuralNet.py  <br />

Thank You <br />
