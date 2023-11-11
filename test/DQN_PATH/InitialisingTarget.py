# Import necessary libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2
import os
import shutil

# Define functions for weight and bias variables, convolution, and batch input retrieval
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def getBatchInput(inputs, start, batchSize):
    first = start
    start = start + batchSize
    end = start
    return inputs[first:end], start

# Define image and training parameters
imageSize = [25, 25, 3]
batchSize = 10 #100
games = 100	#200
# totalImages = games * imageSize[0] * imageSize[1]
#====================
real_pic_width = 1280
real_pic_height = 720
black_area_width   = int(imageSize[0]-real_pic_height/(real_pic_width/imageSize[0]))*imageSize[0]
totalImages = games * (imageSize[0] * imageSize[1] - black_area_width)
# print(totalImages)
# breakpoint()
#====================
learningRate = 0.001
lr_decay_rate = 0.9
lr_decay_step = 2000

# Create a folder for model checkpoints
folderName = 'NewCheckpoints'
if os.path.exists(folderName):
    shutil.rmtree(folderName)
os.mkdir(folderName)

# Specify the checkpoint file path
checkpointFile = 'NewCheckpoints/Checkpoint3.ckpt'

# Define the input placeholder
x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])

# Define the first convolutional layer
W_conv1 = weight_variable([2, 2, 3, 10])
b_conv1 = bias_variable([10])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

# Define the second convolutional layer
W_conv2 = weight_variable([2, 2, 10, 20])
b_conv2 = bias_variable([20])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# Flatten the second convolutional layer
h_conv2_flat = tf.reshape(h_conv2, [-1, 25 * 25 * 20])

# Define the first fully connected layer
W_fc1 = weight_variable([25 * 25 * 20, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# Define the second fully connected layer (output layer)
W_fc2 = weight_variable([100, 2])
b_fc2 = bias_variable([2])
y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

# Define the loss function and optimizer
loss = tf.reduce_mean(tf.square(y_out), 1)
avg_loss = tf.reduce_mean(loss)
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learningRate, global_step, lr_decay_step, lr_decay_rate, staircase=True)
train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

# Create a TensorFlow session
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# Initialize variables
sess.run(tf.initialize_all_variables())

# Read input images
inputs = np.zeros([totalImages, imageSize[0], imageSize[1], imageSize[2]])
print('Reading inputs')
for i in range(totalImages):
    temp = imageSize[0] * imageSize[1] - black_area_width
    #inputs[i] = cv2.imread('trainImages/image_' + str(int(i / temp)) + '_' + str(i % temp) + '.png')
    inputs[i] = cv2.imread('Images_yun_test/image_' + str(int(i / temp)) + '_' + str(i % temp) + '.png')

print('Inputs read')

# Initialize variables for training loop
start = 0
initialTarget = []
iterations = int(totalImages / batchSize)

# Save the initial model checkpoint
save_path = saver.save(sess, checkpointFile)
print("Model saved in file: %s" % save_path)

print('Number of iterations is %d' % iterations)

# Training loop
for i in range(iterations):
    # Get a batch of input data
    batchInput, start = getBatchInput(inputs, start, batchSize)

    # Forward pass to get the output predictions
    batchOutput = sess.run(y_out, feed_dict={x: batchInput})

    # Print progress every 50 iterations
    if i % 25 == 0: #50
        print('%d iterations reached' % i)

    # Collect the output predictions
    for j in range(batchSize):
        initialTarget.append(batchOutput[j])

# Save the final output predictions to a text file
print(start)
np.savetxt('Targets200_New.txt', initialTarget)
