# Continual_Engine_Assignment
Assignment submission , Abhigyan Raj 18JE0013

ABHIGYAN RAJ ( 18JE0013 )

->Model Architecture Overview
For the classification task of the VIS10CAT.txt data , a deep learning based convolutional neural network is used with 7 convolutional layers which in step downscales the Input Image Size ( 3 * 256 * 256 ) by a factor of 2 per convolutional layer . Each Convolution Layer is followed by a batch-normalization layer which helps in regularization followed by ReLU activation. 
The last convolution layer output’s is flattened and fed to a dense layer which outputs the score for each of 10 classes. The Loss function used is Cross Entropy Loss. 


Model Architecture -> model.py
Trained Weights -> './weights'
Test code -> test.py

Report and Instructions to run can be found in -> 'reports'
