from models import *
from utils import *
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input as preprocess
from scipy.spatial.distance import cosine


ROOT = '../1_data_preprocessing/DATA_npy/'


def main():

    test_data1 = np.load(ROOT+'faces_bdr_test.npy')
    test_data2 = np.load(ROOT+'test_texas_vgg.npy')
    labels1 = np.load(ROOT+'labels_bdr_test.npy')
    labels2 = np.load(ROOT+'labels_texas_test.npy')

    test_data = np.concatenate((test_data1,test_data2))
    labels = np.concatenate((labels1,labels2))
    print(test_data.shape)
    print("LOAD MODEL")
    MODEL = resnet('29_persons_relu.h5')

    print("PREDICT ON TEST DATA")
    embeddings = MODEL.predict(test_data)

    #set threshold
    THRESH=0.2

    print("COMPUTE ACCURACY WITH THRESHOLD = ", THRESH,"\n(this should be tuned s.t. FAR=0.001)")

    check_acc(embeddings,labels,THRESH)


    np.save("embeddings_and_labels/test_embeddings.npy",embeddings)
    np.save("embeddings_and_labels/test_labels.npy",labels)


if __name__== "__main__" :
    main()
