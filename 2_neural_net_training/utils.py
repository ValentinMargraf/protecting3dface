import tensorflow as tf
import numpy as np
import random
from models import *
import os
from PIL import Image
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input as preprocess
from scipy.spatial.distance import cosine
from data_augmentation_utils import *



def check_acc(data,labels,THRESH):
    '''
    this function computes FAR, FRR, TMR,
    THRESHold has to be set beforehand, in biometrics it is common use to
    tune this threshold s.t. FAR=0.001
    '''
    num=data.shape[0]
    boolean = np.zeros((num,num))
    ct_far = 0
    ct_frr = 0
    dist_mins = []
    dist_maxs = []
    ct1=0
    ct2=0

    dists=[]
    for i in range(num):
        for j in range(num):
            if j<i:

                v=data[i]
                u=data[j]
                #dist=np.linalg.norm(u-v)
                dist=cosine(u,v)
                if dist <= THRESH and labels[i] == labels[j]:
                    boolean[i,j] = 1
                    ct1+=1
                if THRESH < dist and labels[i] == labels[j]:
                    # this means
                    ct_frr += 1
                if THRESH < dist and labels[i] != labels[j]:
                    boolean[i,j] = 1
                    ct2+=1
                if dist <= THRESH and labels[i] != labels[j]:
                    # this means false accepted
                    ct_far += 1


    num_checks = (num*(num-1))/2
    accuracy = np.sum(boolean)/num_checks
    FAR = ct_far/(ct_far+ct2)
    FRR = ct_frr/(ct_frr+ct1)
    print("Number of Verification Tries: ", num_checks)
    #print("the accuracy is: ", accuracy)
    print("FAR: ",FAR)
    print("FRR: ", FRR)
    print("correctly accepted", ct1)
    print("wrongly denied", ct_frr)
    print("correctly denied", ct2)
    print("wrongly accepted", ct_far)
    print("TMR = 1-FRR", 1 - FRR)





"""
code for the calculation of the loss function is taken from https://github.com/omoindrot/tensorflow-triplet-loss
"""

def _pairwise_cosine(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # squared magnitude of preference vectors (number of occurrences)
    sq_norm = tf.linalg.diag_part(dot_product)
    # inverse squared magnitude
    sq_norm_inv = 1 / sq_norm


    # inverse of the magnitude
    inv_norm = tf.sqrt(sq_norm_inv)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = dot_product * inv_norm
    cosine = tf.transpose(cosine) * inv_norm
    cosine_loss = 1 - cosine
    return cosine_loss

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))



    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0),tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask




def batch_all_triplet_loss(labels, embeddings, margin, cosine=False,squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    if cosine:
        # pairwise cosine similarity matrix
        pairwise_dist = _pairwise_cosine(embeddings)
    else:
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets



def batch_hard_triplet_loss(labels, embeddings, margin, cosine=False, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    if cosine:
        # pairwise cosine similarity matrix
        pairwise_dist = _pairwise_cosine(embeddings)
    else:
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.cast(mask_anchor_positive,tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative,tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


def batch_generator_small(X, y, used_labels, batchsize, img_size, apply_data_augmentation=False):
    '''online batch generator
    applies random cropping and ???'''

    #np.random.seed(0)

    all_indices = np.arange(0,len(used_labels))
    indices = np.random.choice(all_indices, size=batchsize)
    batch=X[indices]
    labels=y[indices]
    batch = np.array(batch)#.reshape(batchsize, img_size[0], img_size[1], img_size[2])
    labels = np.array(labels)
    #print(batch.shape)
    #print(labels.shape)
    if apply_data_augmentation:
        batch, labels = flip_online(batch,labels)
        batch, labels = warp_online(batch,labels)
        #batch, labels = gaussian_noise_online(batch,labels)
    # Shuffling labels and batch the same way
    s = np.arange(batch.shape[0])
    np.random.shuffle(s)

    batch = batch[s]
    labels = labels[s]

    return batch, labels

def batch_generator_array(X, y, used_labels, batchsize, img_size, apply_data_augmentation=False):
    '''online batch generator
    applies random cropping and ???'''

    batch = []
    labels = []


    for l in used_labels:
        all_indices = np.argwhere(y==l).flatten()
        indices = np.random.choice(all_indices, size=batchsize)
        batch.append([X[indices]])
        labels += [l] * batchsize
    batch = np.array(batch).reshape(len(used_labels) * batchsize, img_size[0], img_size[1], img_size[2])
    labels = np.array(labels)
    if apply_data_augmentation:
        batch, labels = flip_online(batch,labels)
        batch, labels = warp_online(batch,labels)
        #batch, labels = gaussian_noise_online(batch,labels)
    # Shuffling labels and batch the same way
    s = np.arange(batch.shape[0])
    np.random.shuffle(s)

    batch = batch[s]
    labels = labels[s]

    return batch, labels


# extract a single face from a given photograph and resize
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
    
def extract_img(filename, required_size=(224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # resize pixels to the model size
    image = Image.fromarray(pixels)
    image = image.resize(required_size)
    IMG = np.asarray(image)
    return IMG
    

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(faces, model):
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess(samples, version=2)
    # perform prediction
    yhat = model.predict(samples)
    return yhat

def load_faces(data_path, size=(224,224)):
    """
    loading the data

    input:
        data_path: path to the data

    output:
        arrays of the images

    """
    files = sorted(os.listdir(data_path), key=lambda x: os.path.getctime(data_path+x))
    n = len(files)
    return np.asarray([extract_face(data_path + '/' + files[i], size) for i in range(n)])

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
