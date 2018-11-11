#!/usr/bin/env python
# coding: utf-8

# ## CNN on CIFAR
# #### By MMA

# In[1]:


import tensorflow as tf


# #### Training Parameter

# In[2]:


LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10


# #### Read Data

# In[3]:


import input_data
cifar = input_data.read_data_sets("cifar-10/", one_hot=True)


# #### Network Parameter

# In[4]:


num_input = 3072
num_classes = 10


# #### Tensorflow Graph Input

# In[5]:


x = tf.placeholder(dtype=tf.float32, shape=(None, num_input), name="input")
y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name="label")


# #### Create some wrappers

# In[6]:


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# In[7]:


def maxpool2d(x, k=2, stride=1):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


# #### Create Model

# In[8]:


def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 32, 32, 3])
    
    # Convolution Layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling
    conv1 = maxpool2d(conv1, k=3, stride=1)
    
    # Convolution Layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling
    conv2 = maxpool2d(conv2, k=3, stride=1)
    
    # Fully Connected Layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1']) + biases['bd1']
    fc1 = tf.nn.relu(fc1)
    
    # Output
    out = tf.matmul(fc1, weights['out']) + biases['out']
    return out


# #### Store Layer Weights and Biases

# In[9]:


weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], 0, 0.01)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], 0, 0.01)),
    'wd1': tf.Variable(tf.truncated_normal([32*32*64, 512], 0, 0.01)),
    'out': tf.Variable(tf.truncated_normal([512, num_classes], 0, 0.01))
}

biases = {
    'bc1': tf.Variable(tf.zeros([64])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bd1': tf.Variable(tf.zeros([512])),
    'out': tf.Variable(tf.zeros([num_classes]))
}


# #### Construct Model

# In[10]:


logits = conv_net(x, weights, biases)
prediction = tf.nn.softmax(logits)


# #### Define loss and optimizer

# In[11]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")


# #### Start Training

# In[12]:


init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    n_batches = int(cifar.train.num_examples / BATCH_SIZE)
    for i in range(EPOCHS):
        for j in range(n_batches):
            batch_x, batch_y = cifar.train.next_batch(BATCH_SIZE)
            with tf.device("/GPU:0"):
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        with tf.device("/GPU:0"):
            loss, acc = sess.run([cross_entropy, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        print("Epoch " + str(i) + ", Minibatch loss= " + "{:.4f}".format(loss) + ", Training Accuracy= "
              + "{:.3f}".format(acc))
    print("Testing Accuracy:",
          sess.run(accuracy_op, feed_dict={x: cifar.test.images[:256], y: cifar.test.labels[:256]}))


# In[ ]:




