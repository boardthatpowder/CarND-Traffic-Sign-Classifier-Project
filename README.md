# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./writeup/histogram.png "Histogram"
[explore]: ./writeup/explore.png "Explore"
[preprocessed]: ./writeup/preprocessed.png "Preprocessed"
[new]: ./writeup/new.png "New"
[new-preprocessed]: ./writeup/new-preprocessed.png "New Preprocessed"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/boardthatpowder/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the distribution of images per class per dataset:

![histogram][histogram]

In addition, I displayed a random set of the images:

![explore][explore]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale by a simple divide by 3 of the individual color channels, along with a simplified normalization of the color channels by subtracting and dividing by 128.  This first draft achieved a validation accuracy of 93%.

However, after reading of the techniques used in the (Traffic Sign Recognition with Multi-Scale Convolutional Networks)[http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] paper, I instead implemented a pre-processing pipeline that consisted of:

- converting the color space of an image from BGR to YUV
- extracting the Y channel
- normalizing the Y channel
- applying Contrast Limited Adaptive Histigram Equalization (CLAHE) to the Y channel

This increased my validation accuracy to 100%!

Here is an example of a traffic sign image before and after preprocessing.

![preprocessed][preprocessed]

Even though my validation accuracy (along with my train and test accuracies) were high, the accuracy of my new images was low at 27% which I put down due to the variations of the images compared to the MNIST images.  To cater for this variance I attempted to implement a data augmentation pipeline that used techniques also described in the (Traffic Sign Recognition with Multi-Scale Convolutional Networks)[http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] paper to create new images by rotating, shifting and zooming images.  Unfortunately I was not able to get this data augmentation step functioning in the timebox I had allocated, therefore is commented out in the Jupyter notebook and not used.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a slight variation of a standard LeNet architecture consisting of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| Batch Normalization | |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| Batch Normalization | |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16 				|
| Flatten | Outputs 400 |
| Fully connected		| Outputs 120 |
| RELU					|												|
| Fully connected		| Outputs 84 |
| RELU					|												|
| Fully connected		| Outputs 43 |

Note that the Batch Normalization should be applied on all the hidden layers, but I was having issues with Batch Normalization after the fully connected layers which I could not resolve within my timebox.
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an:

- Learning rate: 0.001
- Epochs: 30
- Batch size: 128
- Loss operation: Reduced mean of softmax cross entropy with logits
- Optimizer: Adam

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 96.3%
* validation set accuracy of 100%
* test set accuracy of 94.2%
* new test images accuracy of 27%

I took an iterative approach which consisted of:
- Implementing a standard LeNet architecture (as per our previous courseswork)
- Experimentations with learning rate, epochs and batch size
- Experimenting with drop out
- Replacing drop out with batch normalization after reading (Don’t Use Dropout in Convolutional Networks)[https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16]
- Replacing the suggested simple preprocessing techniques with the ones implemented after reading the (Traffic Sign Recognition with Multi-Scale Convolutional Networks)[http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] paper
- Attemopted data augmentation techniques after also reading the (Traffic Sign Recognition with Multi-Scale Convolutional Networks)[http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf] paper

This project was timeboxed, therefore I had ran out of time to finish the batch normalization and data augmentation techniques which I believe would have improved my poor accuracy of the new test images of 27%.  I did not experiment with other architecture types, though this would have been an interesting next step.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 11 German traffic signs that I found on the web:

![new][new]

These are the results of the images after pre-processing:

![new-processed][new-processed]


Most of my images would be difficult to classify when compared to the MNIST dataset due to differences in aspect ratio which would warp the imges when pre-processed to a 32x32 input image.  In addition, the images are much lighter than the MNIST images.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road | Right-of-way at the next intersection  | 
| Yield | Yield	|
| Stop	| Speed limit (30km/h)	|
| Speed limit (30km/h)	| Speed limit (30km/h)	|
| Bumpy road | Slippery road	|
| Slippery road | Slippery road	|
| Road work | Bicycles crossing	|
| Road work | Bicycles crossing	|
| Traffic signals | Right-of-way at the next intersection	|
| Pedestrians | Speed limit (100km/h)	|
| Bicycles crossing | Right-of-way at the next intersection	|

The model was able to correctly guess just 3 of the 11 traffic signs, which gives an accuracy of 27%. This does not compare favorably to the accuracy on the test set of 94.2%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


**Priority road**

The model failed to predict correctly.

Prediction | Probablity
---|---
Right-of-way at the next intersection | 21.1%
Beware of ice/snow | 11.1%
Bicycles crossing | 7.7%
Children crossing | 5.8%
Dangerous curve to the left | 1.4%

**Yield**

The model predicted correctly with the highest accuracy.

Prediction | Probablity
---|---
Yield | 18.9%
Turn right ahead | 5.1%
Turn left ahead | 3.3%
No entry | 2.3%
Ahead only | -1.6%

**Stop**

The model failed to predict correctly, but the correct prediction was 2nd.

Prediction | Probablity
---|---
Speed limit (30km/h) | 15.7%
Stop | 10.7%
Speed limit (80km/h) | 8.2%
Turn left ahead | 6.9%
Speed limit (20km/h) | 1.6%

**Speed limit (30km/h)**

The model predicted correctly, but not with a high accuracy.

Prediction | Probablity
---|---
Speed limit (30km/h) | 5.3%
Speed limit (20km/h) | 4.4%
Speed limit (70km/h) | 2.5%
Roundabout mandatory | -0.3%
Stop | -0.9%

**Bumpy road**

The model failed to predict correctly.

Prediction | Probablity
---|---
Slippery road | 13.1%
Road narrows on the right | 1.8%
Dangerous curve to the right | 0.8%
Children crossing | 0.2%
Bicycles crossing | -0.1%

**Slippery road**

The model predicted correctly, but not with a high accuracy.

Prediction | Probablity
---|---
Slippery road | 12.2%
Bicycles crossing | 9.2%
Beware of ice/snow | 8.0%
Wild animals crossing | -1.7%
Speed limit (50km/h) | -3.1%

**Road work**

The model failed to predict correctly.

Prediction | Probablity
---|---
Bicycles crossing | 12.7%
Speed limit (60km/h) | 9.6%
Dangerous curve to the left | 5.1%
Road narrows on the right | 3.7%
Beware of ice/snow | 1.5%

**Road work**

The model failed to predict correctly.

Prediction | Probablity
---|---
Bicycles crossing | 14.2%
Wild animals crossing | 6.9%
Slippery road | 6.1%
Beware of ice/snow | 6.0%
Bumpy road | 4.5%

**Traffic signals**

The model failed to predict correctly.

Prediction | Probablity
---|---
Right-of-way at the next intersection | 6.3%
Bicycles crossing | 2.0%
Speed limit (30km/h) | -0.0%
Roundabout mandatory | -1.4%
General caution | -2.4%

**Pedestrians**

The model failed to predict correctly.

Prediction | Probablity
---|---
Speed limit (100km/h) | 5.3%
Roundabout mandatory | 4.3%
Go straight or left | 0.9%
Speed limit (50km/h) | -0.6%
Right-of-way at the next intersection | -1.0%

**Bicycles crossing**

The model failed to predict correctly.

Prediction | Probablity
---|---
Right-of-way at the next intersection | 6.9%
General caution | 6.0%
Speed limit (80km/h) | 1.8%
Road work | -1.8%
Bicycles crossing | -2.7%


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


