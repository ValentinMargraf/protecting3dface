import numpy as np
from scipy.spatial.distance import cosine
import pylab
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def compute_match_rates(genuine,impostor,thresh):
    '''
    genuine scores: compared embeddings come from the same person
    impostor: compared embeddings come from different persons
    return FAR, FRR, TMR based on threshold
    '''
    false_accepted = impostor[impostor<thresh].size
    false_rejected = genuine[thresh<genuine].size

    FAR=false_accepted/impostor.size
    FRR=false_rejected/genuine.size
    TMR=1-FRR

    return FAR,FRR,TMR


def get_comparison_scores(data,labels):
    '''
    this function computes genuine and impostor scores
    '''

    output=np.zeros((data.shape[0],3))
    num=data.shape[0]

    genuine=[]
    impostor=[]
    for i in range(num):
        for j in range(num):
            if j<i:

                v=data[i]
                u=data[j]

                dist=cosine(u,v)
                if labels[i] == labels[j]:
                    # compared embeddings come from the same person
                    genuine.append(dist)
                if labels[i] != labels[j]:
                    # compared embeddings come from different persons
                    impostor.append(dist)

    return np.asarray(genuine), np.asarray(impostor)



def sort(V,C_hat,E_hat):
    '''
    this function takes as input an embedding V and its
    belonging parameters C_hat, E_hat.
    the reordered embedding V_sorted will be returned.
    '''
    # compute outer product
    I_hat = np.outer(C_hat,E_hat)
    # perform vec()-operation
    I_hat = I_hat.flatten()%128
    # round to integer
    I_hat = I_hat.astype(int)
    # take first 24 entries
    I_tilde = I_hat[:24]
    indices = np.reshape(I_tilde,(12,2))
    # initialize V'
    V_sorted = np.copy(V)
    for i in range(12):
        V_sorted[[indices[i][0],indices[i][1]]] = V_sorted[[indices[i][1],indices[i][0]]]

    return V_sorted



def generate_params(num_persons,POLY_DEG,SortedRadicals = True):
    '''
    this function generates the parameters C and E for the protection methods
    SortedRadicals and PolyProtect, respectively
    '''
    C=np.zeros((num_persons,POLY_DEG))
    E=np.zeros((num_persons,POLY_DEG))

    # set exponents to choose from for SortedRadicals
    expos=np.array(([1/10,1/9,1/8,1/7,1/6,1/5,1/2,1/3,1/4]))

    for i in range(num_persons):
        # default: SortedRadicals params
        if SortedRadicals:
            b=50001
            a=np.arange(-b+1,b)
            a= a[a != 0]
            inds=np.random.choice(np.arange(POLY_DEG),POLY_DEG,replace=False)
            C[i]=np.random.choice(a,5,replace=False)
            E[i]=np.random.choice(expos,5,replace=False)
        # else: polyprotect coeffs
        if not SortedRadicals:
            E[i] = np.random.choice(np.arange(POLY_DEG)+1,POLY_DEG,replace=False)
            b=51
            a=np.arange(-b+1,b)
            a= a[a != 0]
            C[i] = np.random.choice(a,POLY_DEG,replace=False)


    return C,E


def generate_protected_embeds(V_,C,E,y,output_dim,num_embeddings):

    # compute protected templates
    V=np.hstack((V_,np.zeros((num_embeddings,0))))
    P=np.zeros((num_embeddings,output_dim))
    for i in range(V.shape[0]):
        for j in range(P.shape[1]):
            a=V[i][j:j+5]
            b=np.sign(a) * (np.abs(a))
            P_= b**E[y[i]]
            P[i,j] = C[y[i]]@P_.T
    return P

def generate_protected_embeds_unlink(V_,C,E,y,output_dim,num_embeddings):

    # compute protected templates
    V=np.hstack((V_,np.zeros((num_embeddings,0))))
    P=np.zeros((num_embeddings,output_dim))
    for i in range(V.shape[0]):
        for j in range(P.shape[1]):
            a=V[i][j:j+5]
            b=np.sign(a) * (np.abs(a))
            P_= b**E[i]
            P[i,j] = C[i]@P_.T
    return P

def unlinkability_plot(matedScores,nonMatedScores,num_bins,title):
    '''
    taken from https://github.com/dasec/unlinkability-metric
    '''
    nBins=num_bins
    omega=matedScores.size/nonMatedScores.size
    figureTitle=title

    legendLocation='upper right'
    figureFile = figureTitle

    matplotlib.use('Agg')

    ######################################################################
    ### Evaluation


    if nBins == -1:
        nBins = min(len(matedScores)/10,100)

    # define range of scores to compute D
    bin_edges = np.linspace(min([min(matedScores), min(nonMatedScores)]), max([max(matedScores), max(nonMatedScores)]), num=nBins + 1, endpoint=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2 # find bin centers

    # compute score distributions (normalised histogram)
    y1 = np.histogram(matedScores, bins = bin_edges, density = True)[0]
    y2 = np.histogram(nonMatedScores, bins = bin_edges, density = True)[0]

    # Compute LR and D
    LR = np.divide(y1, y2, out=np.ones_like(y1), where=y2!=0)
    D = 2*(omega*LR/(1 + omega*LR)) - 1
    D[omega*LR <= 1] = 0
    D[y2 == 0] = 1 # this is the definition of D, and at the same time takes care of inf / nan
    # Compute and print Dsys
    Dsys = np.trapz(x = bin_centers, y = D* y1)


    ### Plot final figure of D + score distributions

    plt.clf()

    sns.set_context("paper",font_scale=1.7, rc={"lines.linewidth": 2})
    sns.set_style("white")

    ax = sns.kdeplot(matedScores, label='Genuine', color=sns.xkcd_rgb["medium green"])
    x1,y1 = ax.get_lines()[0].get_data()
    ax = sns.kdeplot(nonMatedScores, label='Impostor', color=sns.xkcd_rgb["pale red"],linewidth=2, linestyle='--')
    x2,y2 = ax.get_lines()[1].get_data()

    ax2 = ax.twinx()
    lns3, = ax2.plot(bin_centers, D, label='$\mathrm{D}_{\leftrightarrow}(s)$', color=sns.xkcd_rgb["denim blue"],linewidth=3)

    # print omega * LR = 1 lines
    index = np.where(D <= 0)
    ax.axvline(bin_centers[index[0][0]], color='k', linestyle='--')

    #index = np.where(LR > 1)
    #ax.axvline(bin_centers[index[0][2]], color='k', linestyle='--')
    #ax.axvline(bin_centers[index[0][-1]], color='k', linestyle='--')


    # Figure formatting
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Score")
    ax.set_title("%s, $\mathrm{D}_{\leftrightarrow}^{\mathit{sys}}$ = %.2f" % (figureTitle, Dsys),  y = 1.02)

    labs = [ax.get_lines()[0].get_label(), ax.get_lines()[1].get_label(), ax2.get_lines()[0].get_label()]
    lns = [ax.get_lines()[0], ax.get_lines()[1], lns3]
    ax.legend(lns, labs, loc = legendLocation)

    ax.set_ylim([0, max(max(y1), max(y2)) * 1.05])
    ax.set_xlim([bin_edges[0]*0.98, bin_edges[-1]*1.02])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("$\mathrm{D}_{\leftrightarrow}(s)$")

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(right=0.88)
    PATH="figs/"+figureFile
    pylab.savefig(PATH,format='pdf')
