
import tensorflow as tf

#Start interactive session
sess = tf.InteractiveSession()


# ### The MNIST data

# In[3]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# ### Initial parameters

# Create general parameters for the model

# In[4]:

width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem


# ### Input and output

# Create place holders for inputs and outputs

# In[5]:

x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])


# #### Converting images of the data set to tensors

# The input image is a 28 pixels by 28 pixels and 1 channel (grayscale)
# 
# In this case the first dimension is the __batch number__ of the image (position of the input on the batch) and can be of any size (due to -1)

# In[6]:

x_image = tf.reshape(x, [-1,28,28,1])  


# In[7]:

x_image


# ### Convolutional Layer 1

# #### Defining kernel weight and bias
# Size of the filter/kernel: 5x5;  
# Input channels: 1 (greyscale);  
# 32 feature maps (here, 32 feature maps means 32 different filters are applied on each image. So, the output of convolution layer would be 28x28x32). In this step, we create a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`

# In[8]:

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs


# 
# #### Convolve with weight tensor and add biases.
# 
# Defining a function to create convolutional layers. To creat convolutional layer, we use __tf.nn.conv2d__. It computes a 2-D convolution given 4-D input and filter tensors.
# 
# Inputs:
# - tensor of shape [batch, in_height, in_width, in_channels]. x of shape [batch_size,28 ,28, 1]
# - a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]. W is of size [5, 5, 1, 32]
# - stride which is  [1, 1, 1, 1]
#     
#     
# Process:
# - change the filter to a 2-D matrix with shape [5\*5\*1,32]
# - Extracts image patches from the input tensor to form a *virtual* tensor of shape `[batch, 28, 28, 5*5*1]`.
# - For each patch, right-multiplies the filter matrix and the image patch vector.
# 
# Output:
# - A `Tensor` (a 2-D convolution) of size <tf.Tensor 'add_7:0' shape=(?, 28, 28, 32)- Notice: the output of the first convolution layer is 32 [28x28] images. Here 32 is considered as volume/depth of the output image.

# In[9]:

convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1



# #### Apply the ReLU activation Function

# In this step, we just go through all outputs convolution layer, __covolve1__, and wherever a negative number occurs,we swap it out for a 0. It is called ReLU activation Function.

# In[10]:

h_conv1 = tf.nn.relu(convolve1)


# #### Apply the max pooling

# Use the max pooling operation already defined, so the output would be 14x14x32

# Defining a function to perform max pooling. The maximum pooling is an operation that finds maximum values and simplifies the inputs using the spacial correlations between them.  
# 
# __Kernel size:__ 2x2 (if the window is a 2x2 matrix, it would result in one output pixel)  
# __Strides:__ dictates the sliding behaviour of the kernel. In this case it will move 2 pixels everytime, thus not overlapping.
# 
# 
# 
# 

# In[11]:

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2


# #### First layer completed

# In[12]:

layer1= h_pool1


# ### Convolutional Layer 2
# #### Weights and Biases of kernels

# Filter/kernel: 5x5 (25 pixels) ; Input channels: 32 (from the 1st Conv layer, we had 32 feature maps); 64 output feature maps  
# __Notice:__ here, the input is 14x14x32, the filter is 5x5x32, we use 64 filters, and the output of the convolutional layer would be 14x14x64.
# 
# __Notice:__ the convolution result of applying a filter of size [5x5x32] on image of size [14x14x32] is an image of size [14x14x1], that is, the convolution is functioning on volume.

# In[13]:


W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs


# #### Convolve image with weight tensor and add biases.

# In[14]:

convolve2= tf.nn.conv2d(layer1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2


# #### Apply the ReLU activation Function

# In[15]:

h_conv2 = tf.nn.relu(convolve2)


# #### Apply the max pooling

# In[16]:

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2


# #### Second layer completed

# In[17]:

layer2= h_pool2


# So, what is the output of the second layer, layer2?
# - it is 64 matrix of [7x7]
# 

# ### Fully Connected Layer 3

# Type: Fully Connected Layer. You need a fully connected layer to use the Softmax and create the probabilities in the end. Fully connected layers take the high-level filtered images from previous layer, that is all 64 matrics, and convert them to an array.
# 
# So, each matrix [7x7] will be converted to a matrix of [49x1], and then all of the 64 matrix will be connected, which make an array of size [3136x1]. We will connect it into another layer of size [1024x1]. So, the weight between these 2 layers will be [3136x1024]
# 
# 
# 

# #### Flattening Second Layer

# In[18]:

layer2_matrix = tf.reshape(layer2, [-1, 7*7*64])


# #### Weights and Biases between layer 2 and 3

# Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64); 1027 outputs to Softmax layer

# In[19]:

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs


# #### Matrix Multiplication (applying weights and biases)

# In[20]:

fcl3=tf.matmul(layer2_matrix, W_fc1) + b_fc1


# #### Apply the ReLU activation Function

# In[21]:

h_fc1 = tf.nn.relu(fcl3)


# #### Third layer completed

# In[22]:

layer3= h_fc1


# In[23]:

layer3


# #### Optional phase for reducing overfitting - Dropout 3

# It is a phase where the network "forget" some features. At each training step in a mini-batch, some units get switched off randomly so that it will not interact with the network. That is, it weights cannot be updated, nor affect the learning of the other network nodes.  This can be very useful for very large neural networks to prevent overfitting.

# In[24]:

keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)


# ###  Layer 4- Readout Layer (Softmax Layer)

# Type: Softmax, Fully Connected Layer.

# #### Weights and Biases

# In last layer, CNN takes the high-level filtered images and translate them into votes using softmax.
# Input channels: 1024 (neurons from the 3rd Layer); 10 output features

# In[25]:

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]


# #### Matrix Multiplication (applying weights and biases)

# In[26]:

fcl4=tf.matmul(layer3_drop, W_fc2) + b_fc2


# #### Apply the Softmax activation Function
# __softmax__ allows us to interpret the outputs of __fcl4__ as probabilities. So, __y_conv__ is a tensor of probablities.

# In[27]:

y_conv= tf.nn.softmax(fcl4)


# In[28]:

layer4= y_conv


# In[29]:

layer4

# # Summary of the Deep Convolutional Neural Network

# So, the basic the structure of  our network

# #### 0) Input - MNIST dataset
# #### 1) Convolutional and Max-Pooling
# #### 2) Convolutional and Max-Pooling
# #### 3) Fully Connected Layer
# #### 4) Processing - Dropout
# #### 5) Readout layer - Fully Connected
# #### 6) Outputs - Classified digits

# ---

# # Define functions and train the model

# #### Define the loss function
# 
# We need to compare our output, layer4 tensor, with ground truth for all mini_batch. we can use __cross entropy__ to see how bad our CNN is working - to measure the error at a softmax layer.
# 
# The following code shows an toy sample of cross-entropy for a mini-batch of size 2 which its items have been classified. You can run it (first change the cell type to __code__ in the toolbar) to see hoe cross entropy changes.
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))
# __reduce_sum__ computes the sum of elements of __(y_ * tf.log(layer4)__ across second dimension of the tensor, and __reduce_mean__ computes the mean of all elements in the tensor..

# In[30]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))


# #### Define the optimizer
# 
# It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) and apply gradients to variables. It will be done by an optimizer: GradientDescent or Adagrad. 

# In[31]:

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# #### Define prediction
# Do you want to know how many of the cases in a mini-batch has been classified correctly? lets count them.

# In[32]:

correct_prediction = tf.equal(tf.argmax(layer4,1), tf.argmax(y_,1))


# #### Define accuracy
# It makes more sense to report accuracy using average of correct cases.

# In[33]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# #### Run session, train

# In[34]:

sess.run(tf.global_variables_initializer())


# *If you want a fast result (**it might take sometime to train it**)*

# In[ ]:

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# <div class="alert alert-success alertsuccess" style="margin-top: 20px">
# <font size = 3><strong>*You can run this cell if you REALLY have time to wait (**change the type of the cell to code**)*</strong></font>
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# _PS. If you have problems running this notebook, please shutdown all your Jupyter runnning notebooks, clear all cells outputs and run each cell only after the completion of the previous cell._

# ---

# <a id="ref9"></a>
# # Evaluate the model

# Print the evaluation to the user

# In[ ]:

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# ## Visualization

# Do you want to look at all the filters?

# In[ ]:

kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32,-1]))


# In[ ]:

from utils import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
get_ipython().magic(u'matplotlib inline')
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  


# Do you want to see the output of an image passing through first convolution layer?
# 

# In[ ]:

import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")


# In[ ]:

ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")


# What about second convolution layer?

# In[ ]:

ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")


# In[ ]:

sess.close() #finish the session


# ### References:
# 
# https://en.wikipedia.org/wiki/Deep_learning    
# http://sebastianruder.com/optimizing-gradient-descent/index.html#batchgradientdescent  
# http://yann.lecun.com/exdb/mnist/  
# https://www.quora.com/Artificial-Neural-Networks-What-is-the-difference-between-activation-functions  
# https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html  
