import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import sys
sys.path.append('../2_neural_net_training/')
from utils import check_acc
from helpers.utils_analysis import *

# default is 10, decide how often you want to run
ITERNUM=10
# set to True if you want to reproduce the numbers stated in the paper
REPRODUCE=True

def main():

    ### SET PARAMS
    POLY_DEG=5
    output_dim = 124
    # True means SortedRadicals parameters (the ones proposed in this paper)
    USE_SORTEDRADICALS = True
    # 29 persons relu, load embeddings train and test
    X=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TEST_relu.npy')
    y=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/LABELS_TEST_relu.npy')
    V=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TRAIN_relu.npy')

    num_embeddings=X.shape[0]

    print("\nACCURACY BASIC")
    check_acc(X, y, 0.2)

    num_persons=len(list(set(list(y))))
    #print("numpersons", num_persons)

    SORTED_genuine=[]
    SORTED_impostor=[]
    RADICALS_genuine=[]
    RADICALS_impostor=[]
    SORTEDRADICALS_genuine=[]
    SORTEDRADICALS_impostor=[]


    for iter_ in range(ITERNUM):

        if REPRODUCE:
            # load stored C and E
            C=np.load("params/C_sr_"+str(iter_)+".npy")
            E=np.load("params/E_sr_"+str(iter_)+".npy")
        else:
            # compute parameters C and E, each iteration a different set
            C,E = generate_params(num_persons,POLY_DEG,USE_SORTEDRADICALS)

        #sort the embeddings
        V_=[]
        for i in range(y.shape[0]):
            Sorted=sort(X[i],C[y[i]],E[y[i]])
            V_.append(Sorted)
        V_=np.asarray(V_)
        SORTED_genuine.append(get_comparison_scores(V_, y)[0])
        SORTED_impostor.append(get_comparison_scores(V_, y)[1])

        # apply nonlinear mapping on unsorted embeddings
        P_rad = generate_protected_embeds(X,C,E,y,output_dim,num_embeddings)
        RADICALS_genuine.append(get_comparison_scores(P_rad, y)[0])
        RADICALS_impostor.append(get_comparison_scores(P_rad, y)[1])

        # apply nonlinear mapping on sorted embeddings
        P_sort_rad = generate_protected_embeds(V_,C,E,y,output_dim,num_embeddings)
        SORTEDRADICALS_genuine.append(get_comparison_scores(P_sort_rad, y)[0])
        SORTEDRADICALS_impostor.append(get_comparison_scores(P_sort_rad, y)[1])


    print("\nACCURACY SORTED")
    if REPRODUCE:
        THRESH=0.3
    else:
        # tune this threshold, s.t. FAR=0.001
        THRESH=0.4
    FAR,FRR,TMR= compute_match_rates(np.asarray(SORTED_genuine).flatten(),np.asarray(SORTED_impostor).flatten(),THRESH)
    print("FAR: ",FAR, "\nFRR: ",FRR,"\nTMR: ",TMR)

    print("\nACCURACY RADICALS")
    if REPRODUCE:
        THRESH=0.42
    else:
        # tune this threshold, s.t. FAR=0.001
        THRESH=0.4
    FAR,FRR,TMR= compute_match_rates(np.asarray(RADICALS_genuine).flatten(),np.asarray(RADICALS_impostor).flatten(),THRESH)
    print("FAR: ",FAR, "\nFRR: ",FRR,"\nTMR: ",TMR)

    print("\nACCURACY SORTED RADICALS")
    if REPRODUCE:
        THRESH=0.45
    else:
        # tune this threshold, s.t. FAR=0.001
        THRESH=0.4
    FAR,FRR,TMR= compute_match_rates(np.asarray(SORTEDRADICALS_genuine).flatten(),np.asarray(SORTEDRADICALS_impostor).flatten(),THRESH)
    print("FAR: ",FAR, "\nFRR: ",FRR,"\nTMR: ",TMR)





if __name__== "__main__" :
    main()
