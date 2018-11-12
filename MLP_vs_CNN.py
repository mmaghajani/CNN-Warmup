#!/usr/bin/env python
# coding: utf-8

# ### MLP vs CNN
# #### By MMA

# ### Fully connected MLP with MNIST

# In[ ]:


import tensorflow as tf


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[4]:


LEARNING_RATE = 0.01
EPOCHS = 1000
BATCH_SIZE = 64


# In[5]:


x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="input")
y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="label")


# In[6]:


with tf.name_scope(name="Hidden_layer_1"):
    w1 = tf.Variable(tf.truncated_normal([784, 500], 0, 0.1), name="W")
    b1 = tf.Variable(tf.zeros([500]), name="B")

    hidden_layer1_input = tf.matmul(x, w1) + b1
    relu1 = tf.nn.relu(hidden_layer1_input)


# In[7]:


with tf.name_scope(name="Hidden_layer_2"):
    w2 = tf.Variable(tf.truncated_normal([500, 500], 0, 0.1), name="W")
    b2 = tf.Variable(tf.zeros([500]), name="B")

    hidden_layer2_input = tf.matmul(relu1, w2) + b2
    relu2 = tf.nn.relu(hidden_layer2_input)


# In[8]:


with tf.name_scope("output_layer"):
    w3 = tf.Variable(tf.truncated_normal([500, 10], 0, 0.1), name="W")
    b3 = tf.Variable(tf.zeros([10]), name="B")

    output_layer_input = tf.matmul(relu2, w3) + b3
    output = tf.nn.softmax(output_layer_input)


# ##### Cross Entropy, Loss Function, Accuaracy, etc.

# In[10]:


with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")


# In[11]:


loss_train_summary_op = tf.summary.scalar("Cross_Entropy", cross_entropy)
acc_train_summary_op = tf.summary.scalar("accuracy_train", accuracy_op)

acc_val_summary_op = tf.summary.scalar("accuracy_validation", accuracy_op)
loss_val_summary_op = tf.summary.scalar("Cross_Entropy_validation", cross_entropy)

filewriter = tf.summary.FileWriter("./graphs_fc")


# ##### Session Run 

# In[12]:


sess_fc = tf.Session()
filewriter.add_graph(sess_fc.graph)

sess_fc.run(tf.global_variables_initializer())


# ##### Learning

# In[14]:


for i in range(EPOCHS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess_fc.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if i % 50 == 0:
        b1, b2 = (sess_fc.run([loss_train_summary_op, acc_train_summary_op],
                              feed_dict={x: batch_xs, y: batch_ys}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)

        batch_xs_v, batch_ys_v = mnist.validation.next_batch(BATCH_SIZE)
        b1, b2 = (sess_fc.run([acc_val_summary_op, loss_val_summary_op],
                              feed_dict={x: batch_xs_v, y: batch_ys_v}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)


# In[15]:


yyy = sess_fc.run(output, feed_dict={x: mnist.test.images})
[accuracy] = sess_fc.run([accuracy_op], feed_dict={output: yyy, y: mnist.test.labels})
print("The accuracy of test:" + str(accuracy))


# In[16]:


filewriter.close()


# #### Cross Entropy in train

# <img src="graphs/fc/cross_entropy_train.png" style="float: left; margin-right: 10px;">

# #### Cross Entropy in validation

# <img src="graphs/fc/cross_entropy_validation.png" style="float: left; margin-right: 10px;">

# #### Accuracy in train

# <img src="graphs/fc/accuracy_train.png" style="float: left; margin-right: 10px;">

# #### Accuracy in validation

# <img src="graphs/fc/accuracy_validation.png" style="float: left; margin-right: 10px;">

# #### Network Graph

# <img src="graphs/fc/graph.png" style="float: left; margin-right: 10px;">

# ### CNN with MNIST

# #### Learning parameters

# In[45]:


LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 1000


# #### Network parameters

# In[46]:


num_inputs = 784 
num_classes = 10


# #### Placeholders and dropout rate

# In[47]:


X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="Inputs")
Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Lables")
keep_prob = tf.placeholder(dtype=tf.float32) # DropOut ( Keep Probability )


# #### Creating some wrappers for convolution and pooling

# In[48]:


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# In[49]:


def maxPool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


# In[50]:


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Layer One
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxPool2d(conv1, k=2)
    # Layer Two
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxPool2d(conv2, k=2)
    # Fully Connected Layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output
    output = tf.add(tf.matmul(fc1, weights['output']), biases['output'])
    return output


# #### Weights and Biases

# In[51]:


weights = {
    # 5*5 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    # 5*5 conv, 64 input, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    # Fully Connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 500])),
    # 500 inputs, 10 outputs ( Class Prediction )
    'output': tf.Variable(tf.random_normal([500, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([500])),
    'output': tf.Variable(tf.random_normal([num_classes])),
}


# #### Model Construction

# In[52]:


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


# #### Define Loss and Optimizer

# In[53]:


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)


# #### Evaluation

# In[54]:


CorrectPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy_op = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))


# In[55]:


loss_train_summary_op = tf.summary.scalar("Cross_Entropy_train", loss_op)
acc_train_summary_op = tf.summary.scalar("accuracy_train", accuracy_op)

acc_val_summary_op = tf.summary.scalar("accuracy_validation", accuracy_op)
loss_val_summary_op = tf.summary.scalar("Cross_Entropy_validation", loss_op)

filewriter = tf.summary.FileWriter("./graphs_cnn")


# #### Session

# In[56]:


sess_cnn = tf.Session()
filewriter.add_graph(sess_cnn.graph)
init = tf.global_variables_initializer()
sess_cnn.run(init)
for i in range(EPOCHS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess_cnn.run(train_op, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
    if i % 50 == 0:
        print(i)
        b1, b2 = (sess_cnn.run([loss_train_summary_op, acc_train_summary_op],
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)

        batch_xs_v, batch_ys_v = mnist.validation.next_batch(BATCH_SIZE)
        b1, b2 = (sess_cnn.run([acc_val_summary_op, loss_val_summary_op],
                               feed_dict={X: batch_xs_v, Y: batch_ys_v, keep_prob: 1.0}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)


# In[57]:


[accuracy] = sess_cnn.run([accuracy_op], feed_dict={X: mnist.test.images[:256],
                                                    Y: mnist.test.labels[:256],
                                                    keep_prob: 1.0})
print("The accuracy of test:" + str(accuracy))
filewriter.close()


# #### Cross Entropy in train

# <img src="graphs/cnn/cross_entropy_train.png" style="float: left; margin-right: 10px;">

# #### Cross Entropy in validation

# <img src="graphs/cnn/cross_entropy_validation.png" style="float: left; margin-right: 10px;">

# #### Accuracy in train

# <img src="graphs/cnn/accuracy_train.png" style="float: left; margin-right: 10px;">

# #### Accuracy in validation

# <img src="graphs/cnn/accuracy_validation.png" style="float: left; margin-right: 10px;">

# #### Network Graph

# <img src="graphs/cnn/graph.png" style="float: left; margin-right: 10px;">

# ## A Comparsion between CNN and fully connected network

# #### Accuracy in FC network was about 70% and in CNN was 98%.

# #### Number of parameters

# In[35]:


print("Number of parameters in FC network : ",
      w1.shape[0]*w1.shape[1] + w2.shape[0]*w2.shape[1] + w3.shape[0]*w3.shape[1] + 
      500 + 500 + 10)


# In[41]:


print("Number of parameters in CNN : ",
      weights['wc1'].shape[0]*weights['wc1'].shape[1]*weights['wc1'].shape[2]*weights['wc1'].shape[3]+
      weights['wc2'].shape[0]*weights['wc2'].shape[1]*weights['wc2'].shape[2]*weights['wc2'].shape[3]+
      weights['wd1'].shape[0]*weights['wd1'].shape[1] + 64 + 64 + 500
      )


# In[ ]:




