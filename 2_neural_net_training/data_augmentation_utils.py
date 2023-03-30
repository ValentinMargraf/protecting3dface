import cv2
#import dlib
import numpy as np
from scipy import ndimage, misc
from imutils import face_utils
import matplotlib.pyplot as plt
import tensorflow as tf
#from pychubby.actions import Multiple, Smile, Chubbify, LinearTransform, OpenEyes, RaiseEyebrow, StretchNostrils, Pipeline
#from pychubby.detect import LandmarkFace


'''
Functions for OFFLINE DATA AUGMENTATION
'''

def rotate(faces_in, labels_in, max_degree=10, how_often=3):
    """
    rotates the incoming image by some random number in range (-max_degree, max_degree)
    how_often specifies the number of repeats of this procedure
    """
    faces_rot = []
    labels_rot = []
    faces = [faces_in]*how_often
    if faces_in.ndim==4:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    elif faces_in.ndim==3:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    for i in range(faces.shape[0]):
        deg = int(np.random.uniform(-max_degree, max_degree))
        faces_rot.append(ndimage.rotate(faces[i], deg, reshape=False, mode='reflect'))
        labels_rot.append(labels[i])
    for i in range(faces_in.shape[0]):
        faces_rot.append(faces_in[i])
        labels_rot.append(labels_in[i])
    return np.asarray(faces_rot), np.asarray(labels_rot)


def flip(faces_in, labels_in, how_often=2):
    """
    flips each face, i.e. reflection
    """
    faces = [faces_in]*how_often
    if faces_in.ndim==4:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    elif faces_in.ndim==3:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    # flip first half of doubled faces
    faces[:faces_in.shape[0]] = np.flip(faces_in,axis=2)
    return np.asarray(faces), np.asarray(labels)


def gaussian_noise(faces_in, labels_in, how_often):
    """
    adds gaussian noise, specified by mu and sigma depending on image range
    """
    faces_in=faces_in.astype(float)
    max_val = faces_in[0].max()
    min_val = faces_in[0].min()
    mu = int(max_val/2)
    faces = [faces_in]*how_often
    if faces_in.ndim==4:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    elif faces_in.ndim==3:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    sigma = max_val/15
    noise = np.random.normal(mu, sigma, (faces.shape))
    imgs = faces+noise
    return imgs, labels


def warp(faces_in, labels_in, how_often):

    faces_warp = []
    labels_warp = []
    faces = [faces_in]*how_often
    if faces_in.ndim==4:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    elif faces_in.ndim==3:
        faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    for i in range(faces.shape[0]):
        if i%1000==0:
            print(i)
        faces_warp.append(tf.keras.preprocessing.image.apply_affine_transform(
        faces[i],theta=int(np.random.uniform(0, 10)),
        tx=0,ty=0,shear=0.1,zx=np.random.uniform(0.9,1),
        zy=np.random.uniform(0.9,1),row_axis=0,col_axis=1,
        channel_axis=2,fill_mode='nearest',cval=-1.0,order=1))
        labels_warp.append(labels[i])
    for i in range(faces_in.shape[0]):
        faces_warp.append(faces_in[i])
        labels_warp.append(labels_in[i])
    return np.asarray(faces_warp), np.asarray(labels_warp)



'''
Functions for ONLINE DATA AUGMENTATION
'''

def flip_online(batch_faces, batch_labels):
    """
    flips random amount of the batch, i.e. reflection
    """
    labels=batch_labels
    faces=batch_faces
    inds = np.arange(batch_faces.shape[0])
    inds_rand=np.random.choice(inds,np.random.randint(0,labels.shape[0],1),replace=False)
    faces[inds_rand] = np.flip(faces[inds_rand],axis=2)
    return np.asarray(faces), np.asarray(labels)


def gaussian_noise_online(batch_faces, batch_labels):
    """
    adds gaussian noise, specified by mu and sigma depending on image range
    """
    batch_faces=batch_faces.astype(float)
    max_val = batch_faces[0].max()
    min_val = batch_faces[0].min()
    mu = int(max_val/2)
    sigma = np.random.uniform(0,(max_val/15))
    noise = np.random.normal(mu, sigma, (batch_faces.shape))
    batch_faces+=noise
    return np.asarray(batch_faces), np.asarray(batch_labels)


def warp_online(batch_faces, batch_labels):

    for i in range(batch_faces.shape[0]):
        batch_faces[i]=tf.keras.preprocessing.image.apply_affine_transform(
        batch_faces[i],theta=int(np.random.uniform(0, 10)),
        tx=0,ty=0,shear=0.1,zx=np.random.uniform(0.9,1),
        zy=np.random.uniform(0.9,1),row_axis=0,col_axis=1,
        channel_axis=2,fill_mode='nearest',cval=-1.0,order=1)

    return np.asarray(batch_faces), np.asarray(batch_labels)






'''
Functions tbd
'''


def crop_out_patches(faces_in, labels_in, how_often, number_patches, patch_size=16, img_size=224):
    """this function randomly crops out patches from
    the given input images in order to make the neural net more robust.

    input:
    - faces: input data, alredy face extracted and resized to (224,224)
    - labels: their corresponding labels
    - how_often: integer, controls number of generated faces
    - number_patches: integer, max number of patches
    - patch_size

    output:
    - more_faces: augmented data
    - more_labels: their labels
    """
    more_faces = []
    more_labels = []
    ct = 0
    faces = [faces_in]*how_often
    faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    for i in range(faces.shape[0]):
        img = faces[i]
        #plt.imshow(img)
        label = labels[i]
        random_number = np.random.randint(number_patches)
        for k in range(random_number):
            x = np.random.randint(patch_size)
            y = np.random.randint(patch_size)
            x_pix = x*int(img_size/patch_size)
            y_pix = y*int(img_size/patch_size)
            # setting = 0 means BLACK
            img[x_pix:x_pix+patch_size,y_pix:y_pix+patch_size,:] = 0
            #plt.imshow(img)
        #print(img[0])
        more_faces.append(img)
        more_labels.append(labels[i])
        ct += 1
    return np.asarray(more_faces), np.asarray(more_labels)

#predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def pychubby(faces_in, labels_in, how_often):
    """
    assumes faces as input, therefore face detector need not to be applied before landmark detector.
    """

    faces_warped = []
    labels_warped = []
    faces = [faces_in]*how_often
    faces = np.asarray(faces).reshape((how_often*faces_in.shape[0],faces_in.shape[1],faces_in.shape[2],faces_in.shape[3]))
    labels = [labels_in]*how_often
    labels = np.asarray(labels).flatten()
    rect = dlib.rectangle(left=0, top=0, right=faces[0].shape[1], bottom=faces[0].shape[0])

    for i in range(faces.shape[0]):
        # !!! THIS CAN FAIL IF LANDMARKS CAN NOT BE ESTIMATED
        # ON DEPTH FACE !!!
        shape = predictor(faces[i], rect)
        shape = face_utils.shape_to_np(shape)
        lf = LandmarkFace(shape,faces[i])

        rand1 = np.random.uniform(-0.1,0.1)   #kleiner 0.2
        rand2 = np.random.uniform(-0.06,0.06) #kleiner 0.1
        rand3 = np.random.uniform(0.95,1.05)  #0.9-1
        rand4 = np.random.uniform(0.95,1.05)  #0.9-1
        rand5 = np.random.uniform(-0.06,0.06) #kleiner 0.1
        rand6 = np.random.uniform(-0.06,0.06) #kleiner 0.1
        rand7 = np.random.uniform(-0.1,0.1)   #kleiner 0.2

        a_s = Smile(rand1)
        a_e = OpenEyes(rand2)
        a_l = LinearTransform(scale_x=rand3, scale_y=rand4)
        a_b = RaiseEyebrow(scale=rand5)
        a_n = StretchNostrils(rand6)
        a_c = Chubbify(rand7)
        a = Pipeline([a_s, a_e, a_l, a_b, a_n, a_c])

        lf_new, df = a.perform(lf)
        faces_warped.append(lf_new.img)
        labels_warped.append(labels[i])

    return np.asarray(faces_warped), np.asarray(labels_warped)
    
    
    
def pychubby_range(range_img, portrait_img, rand):
    """
    assumes faces as input, therefore face detector need not to be applied before landmark detector.
    """

    rect = dlib.rectangle(left=0, top=0, right=range_img.shape[1], bottom=range_img.shape[0])

    shape = predictor(portrait_img, rect)
    shape = face_utils.shape_to_np(shape)
    lf_range = LandmarkFace(shape,range_img)
    lf_portrait = LandmarkFace(shape,portrait_img)

    rand1 = rand[0]#np.random.uniform(-0.1,0.1)   #kleiner 0.2
    rand2 = rand[1]#np.random.uniform(-0.06,0.06) #kleiner 0.1
    rand3 = rand[2]#np.random.uniform(0.95,1.05)  #0.9-1
    rand4 = rand[3]#np.random.uniform(0.95,1.05)  #0.9-1
    rand5 = rand[4]#np.random.uniform(-0.06,0.06) #kleiner 0.1
    rand6 = rand[5]#np.random.uniform(-0.06,0.06) #kleiner 0.1
    rand7 = rand[6]#np.random.uniform(-0.1,0.1)   #kleiner 0.2

    a_s = Smile(rand1)
    a_e = OpenEyes(rand2)
    a_l = LinearTransform(scale_x=rand3, scale_y=rand4)
    a_b = RaiseEyebrow(scale=rand5)
    a_n = StretchNostrils(rand6)
    a_c = Chubbify(rand7)
    a = Pipeline([a_s, a_e, a_l, a_b, a_n, a_c])

    chub_range, df = a.perform(lf_range)
    chub_portrait, df = a.perform(lf_portrait)

    return chub_range.img, chub_portrait.img
