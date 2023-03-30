from scipy.optimize import root
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import sys
sys.path.append('../2_neural_net_training')
from models import check_acc
from helpers.utils_analysis import *


ITERNUM=10
# set to True if you want to reproduce the numbers stated in the paper
REPRODUCE=True

def main():



    X=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TEST_relu.npy')
    y=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/LABELS_TEST_relu.npy')
    V=np.load('embeddings_and_labels/TEXAS_BDR_29_persons_relu_layer/EMBEDDINGS_TRAIN_relu.npy')


    genuine,impostor=get_comparison_scores(X, y)
    numBins = 60
    unlinkability_plot(genuine,impostor,numBins,"Basic")


    num_embeddings=X.shape[0]
    num_persons=len(list(set(list(y))))
    POLY_DEG=5
    output_dim=124

    # mated = genuine
    # nonmated = impostor
    SORTED_genuine=[]
    SORTED_impostor=[]
    RADICALS_genuine=[]
    RADICALS_impostor=[]
    SORTEDRADICALS_genuine=[]
    SORTEDRADICALS_impostor=[]


    for iter_ in range(ITERNUM):

        if REPRODUCE:
            # load stored C and E
            C=np.load("params/unlinkability/C_"+str(iter_)+".npy")
            E=np.load("params/unlinkability/E_"+str(iter_)+".npy")
        else:
            # compute parameters C and E, each iteration a different set
            C,E = generate_params(num_embeddings,POLY_DEG)

        #sort the embeddings
        V_=[]
        for i in range(y.shape[0]):
            Sorted=sort(X[i],C[i],E[i])
            V_.append(Sorted)
        V_=np.asarray(V_)
        SORTED_genuine.append(get_comparison_scores(V_, y)[0])
        SORTED_impostor.append(get_comparison_scores(V_, y)[1])

        # apply nonlinear mapping on unsorted embeddings
        P_rad = generate_protected_embeds_unlink(X,C,E,y,output_dim,num_embeddings)
        RADICALS_genuine.append(get_comparison_scores(P_rad, y)[0])
        RADICALS_impostor.append(get_comparison_scores(P_rad, y)[1])

        # apply nonlinear mapping on sorted embeddings
        P_sort_rad = generate_protected_embeds_unlink(V_,C,E,y,output_dim,num_embeddings)
        SORTEDRADICALS_genuine.append(get_comparison_scores(P_sort_rad, y)[0])
        SORTEDRADICALS_impostor.append(get_comparison_scores(P_sort_rad, y)[1])


    # default of numBins in https://github.com/dasec/unlinkability-metric is nBins = 100, gives extremely plots.
    # we chose to set 30, since less noisy results and number do not change dramatically
    numBins=30
    unlinkability_plot(np.asarray(SORTED_genuine).flatten(),np.asarray(SORTED_impostor).flatten(),numBins,"Sorted")
    unlinkability_plot(np.asarray(RADICALS_genuine).flatten(),np.asarray(RADICALS_impostor).flatten(),numBins,"Radicals")
    unlinkability_plot(np.asarray(SORTEDRADICALS_genuine).flatten(),np.asarray(SORTEDRADICALS_impostor).flatten(),numBins,"SortedRadicals")


if __name__== "__main__" :
    main()
