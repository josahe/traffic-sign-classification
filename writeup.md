# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/visualization.png "Visualization"
[image2before]: ./report_images/colour.png "Grayscaling before"
[image2after]: ./report_images/grayscale.png "Grayscaling after"
[image3]: ./report_images/augmented.png "Augmented"
[image4]: ./report_images/visualization2.png "Visualization after"
[image5]: ./german_traffic_signs/image1_classid2.jpg "Traffic Sign 1"
[image6]: ./german_traffic_signs/image3_classid11.jpg "Traffic Sign 2"
[image7]: ./german_traffic_signs/image5_classid14.jpg "Traffic Sign 3"
[image8]: ./german_traffic_signs/image1_classid2.jpg "Traffic Sign 4"
[image9]: ./german_traffic_signs/image4_classid25.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. A basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is : 34799 examples
* The size of the validation set is : 4410 examples
* The size of test set is : 12630 examples
* The shape of a traffic sign image is : 32x32x3
* The number of unique classes/labels in the data set is : 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images belong to each of the 43 unique classes/labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. A description of preprocessing the image data. 

As a first step, I decided to convert the images to grayscale because it reduces the complexity of the input data and thus benefits from an increase in computational efficiency. It also simplifies the network and any further image processing applied.

Here is an example of a traffic sign image before and after grayscaling. These images belong to class 0.

##### Before
![alt text][image2before]

##### After
![alt text][image2after]

I didn't feel the need to normalise the data as after converting the images to grayscale the pixel values were in the range of -0.5 to 0.5.

I decided to generate additional data because there was a large disparity between the the number of training examples that belonged to each traffic sign class. The intention was to alter the distribution such that there were 3500 images to each class.

To add more data to the the data set, I added rotated versions of the images. I did this by selecting uniquely random angles between -20 to 20 degrees of random images from a given class, until that class contained 3500 images.

Here is an example of an augmented image (again using class 0):

![alt text][image3]

The difference between the original data set and the augmented data set is an equal distribution across the classes

![alt_text][image4]


#### 2. A description of the model architecture.

My model consisted of the following layers:

| Layer         		| Description	        					    | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x6 	|
| RELU					|												|
| Dropout        	   	| 0.75 probability, outputs 26x26x6 			|
| Max pooling	      	| 2x2 stride, outputs 13x13x6 				    |
| Convolution 6x6     	| 1x1 stride, valid padding, outputs 8x8x16 	|
| RELU					|												|
| Dropout        	   	| 0.75 probability, outputs 8x8x16 			    |
| Max pooling	      	| 2x2 stride, outputs 4x4x16 				    |
| Fully connected		| flattened, outputs 120						|
| RELU					|												|
| Dropout        	   	| 0.75 probability, outputs 8x8x16 			    |
| Fully connected		| outputs 84						            |
| RELU					|												|
| Dropout        	   	| 0.75 probability, outputs 8x8x16 			    |
| Fully connected		| outputs 43						            |
| Softmax				|           									|
| Cross entropy			|           									|
| Gradient Descent		| Adam algorithm          						|


#### 3. A description of how I trained the model.

To train the model, I used an optimizer to implement stochastic gradient descent (SGD). The optimisation algorithm chosen was the Adam algorithm, which includes the use of momentum to smooth out (average) the gradient estimations to prevent spurious steps in the wrong direction. The batch size chosen to implement SGD was 32 and the number of epochs ran was 10.

##### Hyperparameters
To initialise the weights and biases, a truncated normal distribution was used with a mean of 0 and a standard deviation of 0.1. 

An initial learning rate of 0.0005 was used, with learning rate decay included to prevent aggressive learning in the later epochs that could cause overfitting.

The last hyperparameter used, also to prevent overfitting, was a dropout of 75% of variable updates during back propagation.

#### 4. A description of the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* validation set accuracy of 95.3%
* test set accuracy of 93.3%

I took the LeNet-5 architecture as my starting point, borrowed from a previous image classification problem. I updated the topology of the graph to fit with the new image shape and then began to test the network.

Initially the architecture was giving me about 75-80% accuracy on my validation set. An indication of underfiting.

The first step was to increase the filter size on the convolution layer. As the images are small and the features in the images make up a large proportion of the image, a smaller filter size may not be as well placed to pick up on these different features. The pooling step was left as is.

As the validation accuracy increased, I included dropout later on in the process to reduce any chance of overfitting.


### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] 


#### 2. A discussion of the model's predictions on the new traffic signs.

Here are the results of the prediction:

| Image			        | Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Right of way 			| Right of way									|
| Stop Sign      		| Stop sign   									| 
| 50 km/h	      		| 100 km/h					 				    |
| Road work				| Road work										|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.3%

#### 3. A description of how certain the model was when predicting the new images.

The code for making predictions on my final model is located in the 17th code cell of the Ipython notebook.

##### Image 1
The model is absolutely sure that this is a stop sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------:| 
| .99         			| Stop    							| 
| .01     				| No entry							|
| .00					| Keep right						|
| .00	      			| Priority road				 		|
| .00				    | Yield      						|

##### Image 2
The model is absolutely sure that this is a right-of-way sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| .99         			| Right-of-way at the next intersection	| 
| .01     				| Beware of ice/snow					|
| .00					| End of speed limit (80km/h)			|
| .00	      			| Double curve		 				    |
| .00				    | Speed limit (60km/h)					|

##### Image 3
The model is absolutely sure that this is a stop sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| .99         			| Stop    									| 
| .01     				| No entry									|
| .00					| Slippery Road 							|
| .00	      			| Turn right ahead				 			|
| .00				    | Keep right	     						|

##### Image 4
The model is sure that this is a speed sign by looking at the top five soft max probabilities, but does not guess the right one:

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| .99         			| Speed limit (100km/h) 					| 
| .01     				| Speed limit (80km/h)					    |
| .00					| Speed limit (50km/h)						|
| .00	      			| Speed limit (60km/h)		 				|
| .00				    | Speed limit (30km/h)			            |

##### Image 5
The model is absolutely sure that this is a road work sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| .99         			| Road work   								| 
| .01     				| Keep left									|
| .00					| Wild animals crossing						|
| .00	      			| Bumpy road				 				|
| .00				    | Road narrows on the right  				|

