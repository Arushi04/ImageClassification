# Image Classification using Pytorch on CIFAR10

<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/cifar.png" width="800" height="300">

### Description :
Image classification is the process of labeling images according to predefined categories. An image classification model is fed a set of 
images within a specific category. Based on this set, the algorithm learns which class the test images belong to, and can then predict the
correct class of future image inputs, and can even measure how accurate the predictions are. In this project, we have trained our model 
using Convolutional Neural Network on CIFAR10 dataset.

### Dataset:
CIFAR10 dataset has been loaded from PyTorch. This dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 


### Relevant Files:
The project is broken down into 4 files:
1. **dataset.py :** Loading and pre-processing the dataset using Data Loader
2. **models.py :** Defining the layers of the CNN network classifier 
3. **main.py :** Training the training dataset using the defined model and predicting classes for test images. Visualizing traing and test loss and accuracy on test datasets
4. **utils.py :** Additional functions are defined for plotting images.


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

### Command to Run:

python main.py --dataset CIFAR10 --outdir output/ --epochlen 10


### Results and Visualization
* CIFAR10 :


<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/train_loss.png" width="450" height="300">
<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/test_loss.png" width="450" height="300">
<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/train-test-accuracy.png" width="450" height="300">
Train : Orange.              
Test : Blue.        

<img src="https://github.com/Arushi04/ImageClassification/blob/master/images/confusion_matrix.png" width="450" height="300">
       





