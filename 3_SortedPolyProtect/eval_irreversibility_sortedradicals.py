from scipy.optimize import root
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from helpers.invert_polyprotect_complex import invert_P_complex, system_complex, system
from helpers.calculate_embeddings import calc_embedding_distributions
from helpers.utils_analysis import *


# set params to get results for table 4 in the paper
# scale means scaling the embeddings to [0,1]
# use_exact_jacobians refer to using exact jacobians
# inside the levenberg marquardt algorithm, always False
# since True would result to numerical division by zero errors
scale = True
use_exact_jacobians = False
# we chose 10 to get meaningful statistical results, takes quite some time though,
# NUM_ITER = 1 already gives a quite meaningful impression of the inversion rates
NUM_ITER=10
# Take same threshold that corresponds to FAR=0.1% for the basic system
THRESH=0.2
# set to True if you want to reproduce the numbers stated in the paper
REPRODUCE=True

def main():


    print("SETTING: \nembeddings scaled: ",scale,"\nexact jacobians: CAN NOT be used, due to parameter choice in our method!")

    # load data
    V=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TEST_relu.npy')
    labels=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/LABELS_TEST_relu.npy')
    X=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TRAIN_relu.npy')
    scale_param=np.max(np.array(([X.max(),V.max()])))
    if scale==True:
        V=V/(scale_param)
        X=X/(scale_param)




    # estimate the train embeddings distribution
    means, mins, maxs, probabilities = calc_embedding_distributions(X, 3)

    el_probabilities=probabilities
    el_maxs=maxs
    el_mins=mins
    num_guesses=100
    precision=3
    cosines=[]
    index_list=[]
    num_persons=len(list(set(list(labels))))
    overlap=4


    for i in range(NUM_ITER):
        if REPRODUCE:
            C=np.load("params/C_sr_"+str(i)+".npy")
            print("params/C_sr_"+str(i)+".npy loaded")
            E=np.load("params/E_sr_"+str(i)+".npy")
            print("params/E_sr_"+str(i)+".npy loaded")
        else:
            # compute parameters C and E, each iteration a different set
            C,E = generate_params(num_persons,5,SortedRadicals=True)

        # compute protected embeddings
        P = generate_protected_embeds(V,C,E,labels,124,V.shape[0])

        for index in range(V.shape[0]):

            P_ = P[index]
            V_ = V[index]
            E_ = E[labels[index]]
            C_ = C[labels[index]]
            sol, cosine = invert_P_complex(overlap, V_, P_, C_, E_, num_guesses, el_mins, el_maxs, el_probabilities, precision)

            cosines.append(cosine)
            print("Cosine distance between found solution V_hat and V_true: ", cosine)
        checkpoint=np.asarray(cosines)
        print((checkpoint[checkpoint<THRESH].shape)[0]/(checkpoint.shape[0]))


    cosines=np.asarray(cosines)
    print("Inversion Success Rate")
    # if the found solution v_hat is close enough according to the threshold, than we consider
    # this a sucessful inversion
    print((cosines[cosines<THRESH].shape)[0]/(cosines.shape[0]))





if __name__== "__main__" :
    main()
