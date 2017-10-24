
# Traffic Sign Recognition
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

[image1]: ./examples/number_per_class.png "Visualization"
[image4]: ./17_no_entry.png "Traffic Sign 1"
[image5]: ./18_general_caution.png "Traffic Sign 2"
[image6]: ./27_pedestrains.png "Traffic Sign 3"
[image7]: ./28_children_crossing.png "Traffic Sign 4"
[image8]: ./3_speed_limit_60km_per_hr.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Cpruce/SDC-TrafficSigns/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed by label frequency. 

!['Number per class'][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first and last step, I normalized the image data. Referencing https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn#185857, normalizing the input is good because doing so will help a learner stay consistent and make long strides in the search optimization.

I did not convert to gray-scale because I believe that color does contain valuable information and that the speed is already pretty quick on my cpu. Converting to gray scale certainly would speed the process up, but probably with a lower potential accuracy level.

I will revist augmentations and metrics if I have the time. However, since I've used random flip, rotate, and crop and have implemented recall, precision, and f1 score before, I feel these are less of a priority for me.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| activation									|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x6 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| activation									|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x16 	|
| Flatten       		| input 5x5x16, outputs 400x1        			|
| Fully connected		| input 400x1, outputs 120x1     				|
| RELU					| activation									|
| Fully connected		| input 120x1, outputs 84x1        				|
| RELU					| activation									|
| Fully connected		| input 84x1, outputs 43x1     					|
| Softmax				| softmax_cross_entropy_with_logits        		|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 100 epochs to start, settling around 55. My batch size fluctuated, but I eventually left it at 128. I kept the optimizer as the Adam optimizer in order to minimize hyperparameters. For learning rate, I found lower was better, though for a short range, at least for the epochs/instances that I experimented with. Only specifying the initial learning rate, I wound up with 0.0009.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ~1.0
* validation set accuracy of 0.951
* test set accuracy of 0.934

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? *LeNet.*
* Why did you believe it would be relevant to the traffic sign application? *I had a bias that the architecture was simple to adapt to the problem in addition to a guaranteed, straightforward solution. Applying Occam's Razor, this seemed like the best approach.*
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? *The training accuracy is a good indicator of the validation accuracy and, likewise, the validation score is a good indicator of the test score. This few percentage point margin of error indicates that the under/over-fitting error is small at the moment.*
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last three images proved to be troublesome. My hypothesis is that the classifier had a difficult time discerning which is which since these classes look like several others, ie. pedestrian -> general caution and speed limit 60km/h -> speed limit 50km/h. The first two are consistently predicted correctly. This isn't surprising as these two are apparently extremes/unique. No entry is similar to no passing. However, the connection to the circle seems to be a dead give-away. The general caution sign is also very unique in that it is triangular and is not to busy with an obvious exlcamation mark in the middle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| General Caution     	| General Caution 								|
| Pedestrian			| Right of way at the next intersection			|
| Children Crossing	    | Road narrows on the right					 	|
| Speed limit 60km/h	| Go straight or right      					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares unfavorably to the accuracy on the test set of 93%. Sometimes the model achieves 60%, or 3 out of 5, accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is merged with the 2nd visualization of the 5 images. 

A direct consequence of overfitting, my model seems very confident in its predictions, all being 1. This is great for the correct predictions, but really bad for wrong guesses. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry  									| 
| 1.0     				| General Caution								|
| 1.0					| Right of way at the next intersection			|
| 1.0	      			| Road narrows on the right						|
| 1.0				    | Go straight or right  						|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Using https://discussions.udacity.com/t/visualize-the-neural-networks-state-with-test-images/307031/6 as referrence, I extracted what layers I found were compatible with the given function. It seems that the network picked up on relative edges and shading for the most part.



