### <i>Convolution Neural Networks for Visual Recognition - Assignment 1</i>

This is the first assignment of Stanford's [CS231n](http://cs231n.stanford.edu/) course. 

#### Overview
In this assignment, images from the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) data set were classified.  

The solutions are contained in an Ipython Notebook with training, prediction and utility algorithms in separate Python source files.

#### Algorithms
The assignment consisted of five parts:
* K-Nearest Neighbor Classifier
* Support Vector Machine Classifier
* Softmax Classifier
* Two-Layer Neural Network Classifier
* Higher Level Representations of Image Features
  - Histogram of Oriented Gradients (HOG)
  - Color histogram using the hue channel in HSV color space.

For each classifier, vectorized training and prediction algorithms were developed, models were trained using cross validation and hyper-parameters were tuned based on validation results. The SVM and Neural network models were further trained on higher level features, comparing accuracy results with those of raw pixel models.  

#### Visualizations
Various graphs were developed to visual the models' operation and performance.  
* Distance matrix for KNN
* Training loss history
* Classification accuracy history
* Hyper parameter influence on accuracy

To gain intuition on the operation of the algorithms, images were plotted to visualize
* Mean-centered features
* Learned weights
* Misclassified samples