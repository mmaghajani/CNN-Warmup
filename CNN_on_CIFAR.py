#!/usr/bin/env python
# coding: utf-8

# ## CNN on CIFAR
# #### By MMA

# **Define Cifar API as MNIST API**

# In[ ]:


import pickle
import numpy as np


class DataSet:
    def __init__(self, data, labels, one_hot=False):
        self.images = np.array(data, ndmin=2)
        if one_hot:
            self.labels = self.__one_hot(labels)
        else:
            self.labels = labels

        self.num_examples = len(data)
        self.__index = 0

    @staticmethod
    def __one_hot(labels):
        new_label = []
        for label in labels:
            row = []
            for i in range(10):
                if i == label:
                    row.append(1)
                else:
                    row.append(0)
            new_label.append(row)
        return np.array(new_label, ndmin=2)

    def next_batch(self, batch_size):
        if self.__index + batch_size > self.images.shape[0]:
            self.__index = 0
        x, y = self.images[self.__index: self.__index + batch_size], self.labels[self.__index: self.__index + batch_size]
        self.__index = (self.__index + batch_size) % self.images.shape[0]
        return x, y

    def normalize(self):
        self.images = np.array(list(map(lambda x: x/255, self.images)))


class Cifar:
    def __init__(self, batches, test, one_hot=False):
        data = []
        labels = []
        for batch in batches:
            for i in range(len(batch[b'data'])):
                data.append(batch[b'data'][i])
                labels.append(batch[b'labels'][i])

        self.train = DataSet(data[:int(4 * len(data) / 5)], labels[:int(4 * len(data) / 5)], one_hot)
        self.validation = DataSet(data[int(4 * len(data) / 5):], labels[int(4 * len(data) / 5):], one_hot)
        self.test = DataSet(test[b'data'], test[b'labels'], one_hot)

    def normalize_data(self):
        self.train.normalize()
        self.validation.normalize()
        self.test.normalize()


def read_data_sets(url, one_hot=False):
    batches = []
    for i in range(1, 6):
        with open(url + 'data_batch_' + str(i), 'rb') as f:
            batches.append(pickle.load(f, encoding='bytes'))
            f.close()
    with open(url + 'test_batch', 'rb') as f:
        test = pickle.load(f, encoding='bytes')
        f.close()
    return Cifar(batches, test, one_hot)





# In[ ]:


import tensorflow as tf
import numpy as np


# #### Training Parameter

# In[ ]:


LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 50000


# #### Read Data

# In[ ]:


cifar = read_data_sets("cifar-10-batches-py/", one_hot=True)
cifar.normalize_data()


# #### Network Parameter

# In[ ]:


num_input = 3072
num_classes = 10


# #### Tensorflow Graph Input

# In[ ]:


X = tf.placeholder(dtype=tf.float32, shape=(None, num_input), name="Inputs")
Y = tf.placeholder(dtype=tf.float32, shape=(None, num_classes), name="Lables")
keep_prob = tf.placeholder(dtype=tf.float32) # DropOut ( Keep Probability )


# #### Create some wrappers

# In[ ]:


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# In[ ]:


def maxpool2d(x, k=2, stride=1):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


# #### Create Model

# In[ ]:


conv1 = None
def conv_net(x, weights, biases, dropout):
    global conv1
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
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    # Output
    output = tf.add(tf.matmul(fc1, weights['output']), biases['output'])
    return output


# #### Store Layer Weights and Biases

# In[ ]:


weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], 0, 0.01)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], 0, 0.01)),
    'wd1': tf.Variable(tf.truncated_normal([32*32*64, 512], 0, 0.01)),
    'output': tf.Variable(tf.truncated_normal([512, num_classes], 0, 0.01))
}

biases = {
    'bc1': tf.Variable(tf.zeros([64])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bd1': tf.Variable(tf.zeros([512])),
    'output': tf.Variable(tf.zeros([num_classes]))
}


# #### Construct Model

# In[ ]:


logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)


# #### Define loss and optimizer

# In[ ]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
regularizers = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) +                tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['output'])
loss = tf.reduce_mean(cross_entropy + 0.01 * regularizers)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")


# **Define Sumarries**

# In[ ]:


loss_train_summary_op = tf.summary.scalar("Loss_train", loss)
acc_train_summary_op = tf.summary.scalar("accuracy_train", accuracy_op)

acc_val_summary_op = tf.summary.scalar("accuracy_validation", accuracy_op)
loss_val_summary_op = tf.summary.scalar("Loss_validation", loss)

filewriter = tf.summary.FileWriter("./graphs_cnn")


# #### Start Training

# In[14]:


init = tf.global_variables_initializer()
sess_cnn = tf.Session()
filewriter.add_graph(sess_cnn.graph)
sess_cnn.run(init)
for i in range(EPOCHS):
    batch_xs, batch_ys = cifar.train.next_batch(BATCH_SIZE)
    sess_cnn.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})
    if i % 200 == 0:
        print(i)
#           acc = sess_cnn.run(accuracy_op, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
#           print("round : ", i , " Acc : ", acc)
        b1, b2 = (sess_cnn.run([loss_train_summary_op, acc_train_summary_op],
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)

        batch_xs_v, batch_ys_v = cifar.validation.next_batch(BATCH_SIZE)
        b1, b2 = (sess_cnn.run([acc_val_summary_op, loss_val_summary_op],
                               feed_dict={X: batch_xs_v, Y: batch_ys_v, keep_prob: 1.0}))
        filewriter.add_summary(b1, i)
        filewriter.add_summary(b2, i)


# In[15]:


[accuracy] = sess_cnn.run([accuracy_op], feed_dict={X: cifar.test.images[:256],
                                                    Y: cifar.test.labels[:256],
                                                    keep_prob: 1.0})
print("The accuracy of test:" + str(accuracy))


# #### Loss in train

# <img src="graphs/cnn_cifar/loss_train.png">

# #### Loss in validation

# <img src="graphs/cnn_cifar/loss_validation.png">

# #### Accuracy in train

# <img src="graphs/cnn_cifar/accuracy_train.png">

# #### Accuracy in validation

# <img src="graphs/cnn_cifar/accuracy_validation.png">

# #### Graph network

# <img src="graphs/cnn_cifar/graph.png">

# In[ ]:




