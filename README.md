# Image Classification using Pytorch

<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/cifar.png" width="800" height="300">

### Description :
Image classification is the process of labeling images according to predefined categories. An image classification model is fed a set of 
images within a specific category. Based on this set, the algorithm learns which class the test images belong to, and can then predict the
correct class of future image inputs, and can even measure how accurate the predictions are. In this project, we have trained our model 
using Convolutional Neural Network.

### List of Datasets
* CIFAR10
* MNIST (coming up)

### Architecture
* Convolution Neural Network

### Requirements
* Python 3.6.10  
* Numpy 1.18.4  
* Tensorboard 2.0.0   
* Pytorch 1.5.0  
* Torchvision 0.6.0 
* Matplotlib 3.2.1
* Scikit-learn 0.23.1


### Results
* CIFAR10 :

Command to run : python main.py --dataset CIFAR10 --outdir output/ --epochlen 10

<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/train_loss.png" width="450" height="300">
<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/test_loss.png" width="450" height="300">
<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/train-test-accuracy.png" width="450" height="300">
Train : Orange.              
Test : Blue.        

<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/confusion_matrix.png" width="450" height="300">
       





