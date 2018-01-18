import scipy.io
import tensorflow as tf
from layers import ConvLayer, InputLayer, MaxPoolLayer, DenseLayer, DropoutLayer
class VGG19():

   
    def define_network(self, image):
        with tf.name_scope("Block1"):
            conv1_1 = ConvLayer(image, 3, 64, name="conv1_1")
            conv1_2 = ConvLayer(conv1_1, 64, 64, name="conv1_2")
            pool1 = MaxPoolLayer(conv1_2, name='pool1')

        with tf.name_scope("Block2"):
            conv2_1 = ConvLayer(pool1, 64, 128, name="conv2_1")
            conv2_2 = ConvLayer(conv2_1, 128, 128, name="conv2_2")
            pool2 = MaxPoolLayer(conv2_2, name='pool2')

        with tf.name_scope("Block3"):
            conv3_1 = ConvLayer(pool2, 128, 256, name="conv3_1")
            conv3_2 = ConvLayer(conv3_1, 256, 256, name="conv3_2")
            conv3_3 = ConvLayer(conv3_2, 256, 256, name="conv3_3")
            conv3_4 = ConvLayer(conv3_3, 256, 256, name="conv3_4")
            pool3 = MaxPoolLayer(conv3_4, name='pool3')

        with tf.name_scope("Block4"):
            conv4_1 = ConvLayer(pool3, 256, 512,  name="conv4_1")
            conv4_2 = ConvLayer(conv4_1, 512, 512, name="conv4_2")
            conv4_3 = ConvLayer(conv4_2, 512, 512, name="conv4_3")
            conv4_4 = ConvLayer(conv4_3, 512, 512, name="conv4_4")
            pool4 = MaxPoolLayer(conv4_4, name='pool4')

        with tf.name_scope("DenseBlock"):
            fc6 = DenseLayer(pool4, 1024, name='fc6')
            drop_6 = DropoutLayer(fc6, dropout_rate=self.p)
            fc7 = DenseLayer(drop_6, 1024, name='fc7')
            drop_7 = DropoutLayer(fc7, dropout_rate=self.p)
        return drop_7