import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from tensorflow.keras import initializers, layers, Input
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from utils import *



class resnet(keras.Model):

    def __init__(self, path_to_load):
        super().__init__()
        self.img_size = 224
        self.num_channels = 3
        self.dropout = 0.5
        self.init_weights = initializers.RandomNormal(mean=0,stddev=0.1)
        self.feature_length = 128
        self.pretrained=True
        self.freeze =True
        self.load_weights=False
        self.create_model(path_to_load)


    def create_model(self, path_to_load):
        inputs = layers.Input((self.img_size, self.img_size, self.num_channels))
        x=inputs
        if self.pretrained:
            x = VGGFace(model='resnet50', include_top=False, input_shape=(self.img_size, self.img_size, self.num_channels), pooling='avg')(x)

        else:
            x = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(self.img_size, self.img_size, self.num_channels), pooling='avg')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(512, activation="relu",kernel_initializer=self.init_weights, name='dense_2')(x)
        #x = Dropout(self.dropout, name='dropout1')(x)
        x = Dense(256, activation="relu",kernel_initializer=self.init_weights, name='dense_3')(x)
        #x = Dropout(self.dropout, name='dropout2')(x)

        output = Dense(self.feature_length, activation="relu", kernel_initializer=self.init_weights, name='dense_out')(x)
        self.model = keras.Model(inputs=inputs, outputs=output)
        for i, layer in enumerate(self.model.layers):
            print(layer.name)
        trainable = False
        for i, layer in enumerate(self.model.layers):
            if i > 1:
                trainable = True
            layer.trainable = trainable
        if self.load_weights:
            print(self.model.summary())
            self.model.load_weights(path_to_load)
            print("loaded weights")
        if not self.freeze:
            trainable = True
            for i, layer in enumerate(self.model.layers):
                layer.trainable = trainable
        self.model.summary()
        #for i, layer in enumerate(self.model.layers):
        #    print(i, "name:",layer.name)
        self.model.load_weights(path_to_load)

    def train_model(self, X, y, used_labels, epochs, batch_size, margin, lr_1, lr_2, path_to_save):
        ADAM_1 = tf.keras.optimizers.Adam(learning_rate=lr_1)
        ADAM = tf.keras.optimizers.Adam(learning_rate=lr_2)
        ##optimizers_and_layers = [(ADAM_1, self.model.layers[:2]), (ADAM_2, self.model.layers[2:])]
        #ADAM = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        self.model.compile(optimizer=ADAM)
        for i in range(epochs):
            print("EPOCH:", i,"/",epochs)
            with tf.GradientTape() as tape:
                data, labels = batch_generator_array(X, y, used_labels, batch_size, (self.img_size, self.img_size, self.num_channels),apply_data_augmentation=False)
                #print("batch shape", data.shape)
                embeds = self.model(data)
                # switch between batch all and batch hard. batch_hard provides faster convergence!
                # batch all i would do at the end. further i first would train with cosine=False (i.e. euclidean distance)
                # and later for finetuning reasons with cosine = True
                #loss, _ = batch_all_triplet_loss(labels, embeds, margin,cosine=False, squared=False)
                loss = batch_hard_triplet_loss(labels, embeds, margin, cosine=True,squared=False)
                print("LOSS: ", loss)
                if (i%99)==0:
                    print("check for range: embed: ",embeds[0])
            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss, self.model.trainable_weights)
            # Update the weights of the model.
            ADAM.apply_gradients(zip(gradients, self.model.trainable_weights))

            if  1.5  > loss and (i%90)==0 and i!=0:
                self.model.save(path_to_save)
                print("model saved")

    def load_weights(self, path_from):
        self.model.load_weights(path_from)

    def predict(self, data):
        return self.model.predict(data)
