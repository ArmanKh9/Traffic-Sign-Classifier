
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[.\Traffic-Sign-Classifier\pics]: # (Image References)

[image1]: .\pics\p_1 "Traffic Sign 1"
[image2]: .\pics\p_2 "Traffic Sign 2"
[image3]: .\pics\p_3 "Traffic Sign 3"
[image4]: .\pics\p_4 "Traffic Sign 4"
[image5]: .\pics\p_5 "Traffic Sign 5"

## Rubric Points


Here is a link to my [Traffic-Sign-Classifier](https://github.com/ArmanKh9/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is one of the images within the data set.

![Sample Image][.\Traffic-Sign-Classifier\pics\Sample_Image.png]

###Design and Test a Model Architecture

####1. Preprocessing
All the three data set pixel values were normalized around zero with sigma of 0.5. This helped avoiding very large values in training layers to be carried over to logits. Large values cause computational error and also it is favorable to keep values around zero with the same sigma in all direction. This results in less search for answer and quicker convergence.


Here is an example of a traffic sign image after normalization.

![Sample Norm][.\Traffic-Sign-Classifier\pics\Sample_Norm.png]



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers with all weights initialized with mean of zero and sigma of 0.2:

Layer 1: Convolutional layer
--Input: 32x32x3 RGB image with normalized pixel values 
--Convolution 3x3 filter, depth 7, 1x1 stride, valid padding, outputs 30x30x7
--Activation Function: RELU
--Max pooling: 2x2 Ksize, 2x2 stride,  outputs 15x15x7

Layer 2: Convolutional layer
--Input: 15x15x7
--Convolution 4x4 filter, depth 36, 1x1 stride, valid padding, outputs 12x12x36
--Activation Function: RELU
--Max pooling: 2x2 Ksize, 2x2 stride,  outputs 6x6x36

Layer 3: Convolutional layer
--Input: 6x6x36
--Convolution 3x3 filter, depth 48, 1x1 stride, valid padding, outputs 4x4x48
--Activation Function: RELU
--No Max Pooling

Flattening layer 3
--Input: 4x4x48
--Output: 768

Layer 4: Fully connected layer
--Input: 768
--Output: 296
--Activation Function: RELU

Layer 5: Fully connected layer
--Input: 296
--Output: 98
--Activation Function: RELU

Layer 6: Fully connected layer
--Input: 98
--Output: 43
--No activation function


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used softmax to calculate probablity of logits and compare against the actual labels. Then reduce mean function used to calculate the loss. Adamoptimizer function was used to propogate back the loss with the specified learning rate. The following hyperparameters were used to run the training.

Learning rate: 0.0048
Epochs: 50
Batch size: 264

####4. I created a table of the hyperparameters with CNN layers in order to keep track of the result when changing the hyperparameters. I started with LeNet approach in LeNet lab practice. I managed the hyperparameters by only varying one parameter at a time to find the optimized value for that parameter. Some parameters have effect on each other but it is easier to find the optimized range of values for one parameter and then start exploring the interactions between multiple parameters. With this approach, I found the optimized value for each of the hyper parameters along with filter sizes and number of layers. I found out that with only one preprocess step and without using drop out method, epochs number has to be large in order to achieve the required accuracy. In my case it was 50 epochs. Also it was partly because of small learning rate that I used.

My final model results were:
* training set accuracy of: not printed
* validation set accuracy of 0.93 to 0.95
* test set accuracy of 0.924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? LeNet because it was performing a very similar task

* What were some problems with the initial architecture? No pre-process, very low accuracy, fluctuating accuracy in epochs

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. Described above

* Which parameters were tuned? How were they adjusted and why? After finding the optimized values for epochs, learning rate and batch size I started to changing the filter sizes to achieve a better accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?The most important choice was to exclude the max pooling layer after the third convolutional layer. This significantly improved the accuracy. Three convolution layer seemed enough for this data set because photos and labels were not complicated. The feautures that the CNN had to identify in each layer were simple and additional convolutional layers caused reduction in accuracy. Unfortunately I had no time to try drop out but I am assuming it could improve the accuracy or at least reduce the training time.

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? because it classifies hand written numbers which is a very similar task to traffic sign classification except that the later has more features to be identified through out the CNN
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? I tried the prediction of the model on 5 images that downloaded from web and it predicted all of the correctly
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
[image1]: .\pics\p_1 "Traffic Sign 1" Stop sign: text inside the sign might be the hardest feature to be identified
[image2]: .\pics\p_2 "Traffic Sign 2" 30 km/hr: text inside the sign might be the hardest feature to be identified
[image3]: .\pics\p_3 "Traffic Sign 3" 50 km/hr: text inside the sign might be the hardest feature to be identified
[image4]: .\pics\p_4 "Traffic Sign 4" Roundabout Mandatory: tip of the arrows and their direction might be the hardest features to be identified
[image5]: .\pics\p_5 "Traffic Sign 5" 120 km/hr: text inside the sign might be the hardest feature to be identified


The first image might be difficult to classify because: not sure what is the answer to this

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


The accuracy of the predictions were 100% after pre-processing the 5 images. Without pre-process, it was 80%
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 30 km/hr     			| 30 km/hr 										|
| 50 km/hr				| 50 km/hr										|
| Roundabout mandatory 	| Roundabout mandatory  		 				|
| 120 km/hr      		| 120 km/hr         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100% after pre-processing the 5 images. This compares favorably to the accuracy on the test set of 0.93 to 0.95

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 3 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the In[76] of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1,0), and the image does contain a stop sign. The top three soft max probabilities were

Image 1:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .44         			| Stop sign   									| 
| .30     				| Right-of-way at the next intersection 		|
| .18					| Yield											|

Image 2:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .212         			| Speed limit (30km/h)   						| 
| .99     				| Speed limit (50km/h)                   		|
| .81					| Speed limit (80km/h)							|

Image 3:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| Speed limit (50km/h)   						| 
| .61     				| Speed limit (30km/h)                   		|
| .45					| Speed limit (80km/h)							|

Image 4:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .131         			| Roundabout mandatory   						| 
| .68     				| Go straight or left                   		|
| .56					| Vehicles over 3.5 metric tons prohibited		|

Image 5:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .180         			| Speed limit (120km/h)   						| 
| .88     				| Speed limit (70km/h)                      	|
| .86					| Speed limit (20km/h)                   		|




