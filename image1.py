import os
import matplotlib.pylab as plt
import matplotlib.image as img
import numpy as np
import scipy
from scipy import misc
image_train_labels = []
image_train_info = []
image_test_labels = []
image_test_info = []

#reshape image to 3D tensor
def _convert_to_RGB_and_resize(x, BGR=False):
    """

    :param x: array returned by .imread(). Note: OpenCV's cv2.imread() returns BGR image!
    :return: RGB image
    """
    if len(x.shape) == 3:
        if x.shape[2] > 3:  # eg. RGBA image
            x = x[:, :, :3]
        elif x.shape[2] == 2:  # greyscale-alpha image
            grey = x[:,:,0]
            x = np.dstack((grey, grey, grey))

    elif len(x.shape) == 2:  # greyscale image
        x = np.dstack((x, x, x))

    if BGR:
        x = x[:, :, ::-1]

    #x = scipy.misc.imresize(x, size=resize_shape)

    return x

for root, dirname, file_name in os.walk("/home/phuong/projects/data/OK"):
    for name in file_name:
        image_train_labels.append([1,0])
        image = _convert_to_RGB_and_resize(img.imread(os.path.join(root, name)))
        image_train_info.append(scipy.misc.imresize(image, (28,28)))
        if(len(image_train_labels) > 300):
            image_test_labels.append([1,0])
            image_test_info.append(scipy.misc.imresize(image,(28,28)))
            if (len(image_test_labels) > 100):
                break

for root, dirname, file_name in os.walk("/home/phuong/projects/data/NG"):
    for name in file_name:
        image_train_labels.append([0,1])
        image =  _convert_to_RGB_and_resize(img.imread(os.path.join(root, name)))
        image_train_info.append(scipy.misc.imresize(image,(28,28)))
        if (len(image_train_labels) > 600):
            image_test_labels.append([0,1])
            image_test_info.append(scipy.misc.imresize(image,(28,28)))
            if (len(image_test_labels) > 200):
                break


import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, [None, 28, 28, 3])

y_ = tf.placeholder(tf.float32, [None,2])
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#first convolution layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

#reshape x to a 4d tensor
#x_image = tf.reshape(x, [-1, 224, 224, 3])

#apply ReLu function
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#second convolution layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#fully connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) #reshape to vector
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#add layer, softmax
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#train model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def shuffle(a, b):
    permutation = np.random.permutation(b.shape[0])
    #print permutation
    shuffled_a = [a[i] for i in permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def make_batch_iterator(x, y, bsize):
    # type: (object, object, object) -> object
    x,y = shuffle(x,y)
    n_batches = len(x) / bsize
    for b in range(n_batches):
        return x[b*bsize: (b+1)*bsize], y[b*bsize: (b+1)*bsize]
    # print type(batch_iterator())
#print image_train_info[0]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(600):
        batch_iterator = make_batch_iterator(image_train_info, np.array(image_train_labels), 50)
        #print np.array(list(batch_iterator)).shape
        #x_val = np.zeros(shape=(50, 28, 28, 3))
        #print x_val[0].shape, len(list(batch_iterator))
        #for batch in list(batch_iterator):
        #for i in range(50):
            #x_val[i] = list(batch_iterator)[0][i]
            #print list(batch_iterator)[0][i].shape, i
            #print batch[0].shape
        train_accuracy = sess.run([train_step, accuracy], feed_dict={x: list(batch_iterator)[0], y_: list(batch_iterator)[1], keep_prob:.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session = sess, feed_dict={x: list(batch_iterator)[0], y_: list(batch_iterator)[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        # train_step.run(session = sess, feed_dict = {x: batch_iterator[0], y_: batch_iterator[1], keep_prob: 0.5})
    batch = make_batch_iterator(image_test_info, np.array(image_test_labels), 200)
    #print batch[0]
    print('test accuracy %g' % accuracy.eval(session = sess, feed_dict={x: list(batch)[0], y_: list(batch)[1], keep_prob: 1.0}))

    #Saving variables
    v1 = tf.get_variable("v1", shape=[3], initializer= tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape = [5], initializer= tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)
    #Add an op to initialize the variables
    init_op = tf.global_variables_initializer()

    #Add ops to saver restore all the variables
    saver = tf.train.Saver()

    #Later, launch the model, initialize the variables, do some work, and save the variables to disk
    with tf.Session() as sess:
        sess.run(init_op)
        #Do some work with the model.
        inc_v1.op.run()
        dec_v2.op.run()
        #save the variables to disk
        save_path = saver.save(sess, "/home/phuong/PycharProjects/model.ckpt")
        print("Model saved in file: %s" % save_path)

    #Restoreing variables
    #    tf.reset_default_graph()
    #Create some variables.
        v1 = tf.get_variable("v1", shape=[3])
        v2 = tf.get_variable("v2", shape=[5])
    #Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "/home/phuong/PycharProjects/model.ckpt")
        print ("Model restored.")
        #Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2: %s" % v2.eval())
