import os
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from pathlib import Path, PurePath
import sys
sys.path.append('../2_neural_net_training/')
from data_augmentation_utils import *
from models import *
#import dlib
import cv2
import csv
from keras_vggface.utils import preprocess_input as preprocess





# extract a single face from a given photograph
def extract_face(fn_portrait, fn_range, i,required_size=(224, 224)):
    # load image from file
    portrait_ = cv2.imread(fn_portrait)
    portrait_ = cv2.cvtColor(portrait_, cv2.COLOR_BGR2RGB)
    #portrait_ = cv2.resize(portrait_,(720,960))
    range_ = cv2.imread(fn_range)
    #range_= cv2.resize(range_,(720,960))
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(portrait_)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = range_[y1:y2, x1:x2]
    # resize to the model size
    face_array = cv2.resize(face,(required_size))
    #print(i)
    return face_array

def load_depth_faces(data_path_portrait, data_path_range, size=(224,224)):
    """
    loading the data

    input:
        data_path: path to the data

    output:
        arrays of the images

    """
    ROOT = "../DATA_depth_images_raw/opensource_DATA/texas/PreprocessedImages"
    n = len(data_path_range)
    #print(n)
    return np.asarray([extract_face(ROOT + '/' + data_path_portrait[i], ROOT + '/' + data_path_range[i], i,size) for i in range(n)])



def main():


    ROOT = "../DATA_depth_images_raw/opensource_DATA/texas/PreprocessedImages"
    files = sorted(os.listdir(ROOT))

    # we have to render depth (=range) and portrait to afterwards apply the face detector
    # to crop the face (since this is only applicable onto RGB not depth faces)
    filenames_range=[]
    filenames_portrait=[]
    for i in range(len(files)):
        p=PurePath(files[i])
        name=p.stem
        #print(p)
        if str(name)[-5:] == "Range":
            filenames_range.append(name+p.suffix)
        else:
            filenames_portrait.append(name+p.suffix)


    data = load_depth_faces(filenames_portrait,filenames_range)

    labels = []
    with open('texas_info.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in filereader:
            label=row[0][-29:-26]
            label=label.replace("g","")
            label=label.replace(",","")
            label=int(label)
            labels.append(label)

    labels=np.asarray(labels)


    test_inds = []
    train_inds = []

    # pay attention here:
    # it may happen that Labels is not in the form np.arange(0,labels.max())

    for j in set(labels):
        indices = np.argwhere(labels==j).flatten()
        occ=len(indices)
        if occ > 17:
        # WE ONLY TAKE IMAGES FROM IDENTITIES, FROM WHICH THERE EXIST AT LEAST 18 IMAGES
            half=9
            #print("HALF",half)
            for i in range(9):
                #print("I",i)
                train_inds.append(indices[i])
            for k in range(9,18):
                #print("K",k)
                test_inds.append(indices[k])
        #print("label",j,"occurs",occ,"times")

    train_inds=np.asarray(train_inds)
    test_inds=np.asarray(test_inds)


    # split data into train/test
    train_texas=data[train_inds]
    test_texas=data[test_inds]



    # split labels into train/test
    labels_texas_train=labels[train_inds]
    labels_texas_test=labels[test_inds]
    np.save("DATA_npy/labels_texas_test.npy",labels_texas_test)




    train_texas = np.asarray(train_texas, 'float32')
    train_texas_vgg = preprocess(train_texas, version=2)

    test_texas = np.asarray(test_texas, 'float32')
    test_texas_vgg = preprocess(test_texas, version=2)
    np.save('DATA_npy/test_texas_vgg',test_texas_vgg)

    print("START DATA AUGMENTATION")
    # flip
    faces_train_vgg_flip, labels_flip_vgg = flip(train_texas_vgg, labels_texas_train)
    print("flipped")
    # rotate
    faces_train_vgg_rot, labels_rot_vgg = rotate(faces_train_vgg_flip, labels_flip_vgg, max_degree=10, how_often=2)
    print("rotated")
    # warp
    faces_train_vgg_warped, labels_vgg_warped = warp(faces_train_vgg_rot, labels_rot_vgg,  how_often=2)
    print("warped")
    # add noise
    faces_train_vgg_noise, labels_vgg_noise = gaussian_noise(faces_train_vgg_rot, labels_rot_vgg,  how_often=1)
    print("added noise")
    # concatenate all and save
    faces_train_vgg_all = np.concatenate((faces_train_vgg_noise,faces_train_vgg_rot))
    labels_train_vgg_all = np.concatenate((labels_vgg_noise,labels_rot_vgg))
    np.save('DATA_npy/TEXAS_AUGMENTED_faces',faces_train_vgg_all)
    np.save('DATA_npy/TEXAS_AUGMENTED_labels',labels_train_vgg_all)
    print("DONE")

if __name__== "__main__" :
    main()
