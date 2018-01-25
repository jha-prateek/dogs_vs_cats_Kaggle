import tflearn
from  tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import os

model_name = "dogsvscats_6_conv_layers"

img_size = 50

tf.reset_default_graph()

convnet = input_data(shape=[None, img_size, img_size, 1], name="input")

convnet = conv_2d(convnet, 32, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 64, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 128, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 128, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 64, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = conv_2d(convnet, 32, 4, activation="relu")
convnet = max_pool_2d(convnet, 4)

convnet = fully_connected(convnet, 1024, activation="relu")
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer="adam", learning_rate=0.001,
                     loss="categorical_crossentropy", name="targets")

model = tflearn.DNN(convnet, tensorboard_dir="log")

if(os.path.exists("{}.meta".format(model_name))):
    model.load(model_name)
    print("MODEL-LOADED")