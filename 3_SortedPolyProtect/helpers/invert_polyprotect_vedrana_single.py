""" Includes several functions used to evaluate the irreversibility of PolyProtect.

Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
Written by Vedrana Krivokuca Hahn <vkrivokuca@idiap.ch>

This file is part of bob.paper.polyprotect_2021.

bob.paper.polyprotect_2021 is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

bob.paper.polyprotect_2021 is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with bob.paper.polyprotect_2021. If not, see
<http://www.gnu.org/licenses/>. """




import numpy
import numpy as np
import math
from scipy.optimize import root
from scipy.spatial.distance import cosine
#from pwkit.lmmin import Problem, ResidualProblem


#from bob.bio.base.score.load import split_four_column
#from bob.measure import far_threshold

def compute_jacobi(v, *data):
    """
    this function was added by Valentin Margraf
    """
    # computes the jacobi matrix of the function for better reliability of levenberg marquardt
    overlap, C, E, P, n = data
    m = len(C) # number of embedding elements used to generate each PolyProtected element
    step_size = m - overlap
    # print("step size: %d" % (step_size))
    decimal_remainder, integer = math.modf((n - m) / step_size)
    # print("decimal remainder: %f" % (decimal_remainder))
    if decimal_remainder > 0:
        padding = math.ceil((1 - decimal_remainder) * step_size)
    else:
        padding = 0
    v = numpy.pad(v, (0, padding), 'constant', constant_values = (0, 0)) # pad v by "padding" zeros at the end, to make it divisible by m, considering the overlap
    height = v.shape[0]

    # initialize with padding size
    jacobi = np.zeros((height,height))

    starting_indices = list(range(0, len(v) - m + 1, step_size))
    storage_ind = 0
    # derivative rule for polynomials
    for ind in starting_indices:
        # print(ind)
        final_ind = ind + m
        # print(final_ind)
        crnt_word = v[ind : final_ind]
        entries=C * E * crnt_word ** (E-1)
        jacobi[storage_ind,ind:final_ind] = entries
        storage_ind = storage_ind + 1
    #print("f result of system",f)
    # only 128x128 contains relevant derivations,
    # rest is 0
    jacobi=jacobi[:n,:n]
    return jacobi




def system(v, *data):
    """ Specifies the system of (nonlinear) equations used to map an embedding to its PolyProtected template.

    **Inputs:**

    v : 1D numpy array of floats
        Contains the UNKNOWN embedding values v_1 ... v_n.

    *data contains:

        overlap : int
            The amount of overlap between sets of embedding elements used to generate each PolyProtected element (0, 1, 2, 3, or 4).

        C : 1D numpy array of integers
            Contains the m user-specific coefficients used for the mapping.

        E : 1D numpy array of integers.
            Contains the m user-specific exponents used for the mapping.

        P : 1D numpy arrray of scalar floats
            The PolyProtected template.

        n : int
            The dimensionality of the original embedding.

    **Outputs:**

    f : 1D numpy array of floats
        The evaluated function.

    """

    # Unpack data arguments:
    overlap, C, E, P, n = data
    # Set up our system of non-linear equations in n unknowns:

    # see ~/.local/lib/python3.9/site-packages/scipy/optimize/_minpack_py.py
    # we take M=n=128, since  input vector length (128 v_i) is not allowed to exceed
    # output vector length (# of equations to be solved)
    # will contain an underdetermined system of equations

    m = len(C) # number of embedding elements used to generate each PolyProtected element
    step_size = m - overlap
    # print("step size: %d" % (step_size))
    decimal_remainder, integer = math.modf((n - m) / step_size)
    # print("decimal remainder: %f" % (decimal_remainder))
    if decimal_remainder > 0:
        padding = math.ceil((1 - decimal_remainder) * step_size)
    else:
        padding = 0
    v = numpy.pad(v, (0, padding), 'constant', constant_values = (0, 0)) # pad v by "padding" zeros at the end, to make it divisible by m, considering the overlap
    starting_indices = list(range(0, len(v) - m + 1, step_size))
    storage_ind = 0
    # print('****')
    f = numpy.zeros(n)
    #print("startind indices", starting_indices)
    for ind in starting_indices:
        # print(ind)
        final_ind = ind + m
        # print(final_ind)
        crnt_word = v[ind : final_ind]
        # print(crnt_word)

        # das jetzt in complex!!!
        f[storage_ind] = (C @ crnt_word ** E).sum() - P[storage_ind]
        storage_ind = storage_ind + 1

    return f



def invert_P(overlap, V, P, C, E, num_guesses, el_mins, el_maxs, el_probabilities, precision,use_exact_jacobians=True):
    """ Tries to invert the PolyProtected template, P, to recover the original embedding, V, using Python's numerical solver "root".

    **Inputs:**

    overlap : int
        The amount of overlap between sets of embedding elements used to generate each PolyProtected element (0, 1, 2, 3, or 4).

    V : 1D numpy array of floats
        The input embedding.

    P : 1D numpy array of floats
        The PolyProtected template.

    C : 1D numpy array of integers
        The coefficients used to generate the PolyProtected template.

    E : 1D numpy array of integers
        The exponents used to generate the PolyProtected template.

    num_guesses : scalar integer
        Number of guesses we should make for the embedding elements (V_1, ..., V_n), for the numerical solver.

    el_mins: 1D numpy array
        The minimum value of each embedding element's distribution.

    el_maxs: 1D numpy array
        The maximum value of each embedding element's distribution.

    el_probabilities : 2D numpy array
        The probabilities representing the histogram bins of each embedding element's distribution.

    precision : int
        The number of decimal places to which embedding elements should be rounded when calculating the possible values for the guesses.

    **Outputs:**

    solution_found : int
        1 if the solver found a solution, otherwise 0.

    cosine_distance : float
        The cosine distance between v and V for the found solution, v.

    """

    # print(P)
    # print(C)
    # print(E)

    # Get the embedding dimension:
    #n = numpy.count_nonzero(V) # count_nonzero() to remove any padding added to V when generating P
    n=128
    #print(n)

    # Define the step size for the guess values:
    step_size = 1.0 * (10 ** -precision)

    # Generate num_guesses guesses for the n embedding elements, with a precision equivalent to the precision used for the histograms used to represent the embedding element distributions:
    guesses = numpy.zeros([n, num_guesses]) # 1 row per embedding dimension, 1 column per guess
    # print('Generating %d guesses ...' % (num_guesses))
    for el_idx in range(n): # For each embedding element (dimension)
        # Get the possible values from its corresponding histogram parameters
        min_value = el_mins[el_idx]
        max_value = el_maxs[el_idx]
        #possible_values = numpy.around(numpy.linspace(min_value, max_value + step_size, num_guesses), decimals = precision)
        possible_values = numpy.around(numpy.arange(min_value, max_value + step_size, step_size), decimals = precision)

        # Generate num_guesses guesses for this particular embedding element (dimension)
        probabilities = el_probabilities[el_idx]
        #print(possible_values.shape)
        #print("do they match?", probabilities.shape)
        crnt_guesses = numpy.random.choice(possible_values, size = [1, num_guesses], replace = True, p = probabilities) # no check for repeated set of guesses, because same guess for current element along with a different guess for another element may converge differently
        #crnt_guesses = numpy.random.choice(possible_values, size = [1, num_guesses], replace = True)
        guesses[el_idx, :] = crnt_guesses
    #guesses=np.array(guesses,dtype=complex)


    # For each set of guesses, try to solve for v_1 ... v_n (i.e., try to recover the n elements of the input embedding, V):
    solution_found = 0
    cosine_distance = float('NaN')
    # print('Solving ...')
    for guess_idx in range(num_guesses): # for each of our num_guesses systems of n equations
        guesses_set = guesses[:, guess_idx] # current set of n guesses
        data = (overlap, C, E, P, n)

        #set func which computes jacobi matrix
        if use_exact_jacobians:
            jac = compute_jacobi
        if not use_exact_jacobians:
            jac=None

        sol = root(system, guesses_set, args = data, method = 'lm', jac=jac) # this is the generated solution for the n
        if sol.success == 1:
            #print('Solution found!')
            solution_found = 1
            cosine_distance = cosine(sol.x, V) # my use of cosine
            check=cosine(V,guesses_set)
            break # no need to look for other solutions (i.e., as far as attacker is concerned, solution has been found)
        elif sol.success != 1:
            continue

    return sol, cosine_distance




def invert_multiple_Ps(f, overlap, Vs, Ps, Cs, Es, num_guesses, el_mins, el_maxs, el_probabilities, precision):
    """ Tries to invert the PolyProtected embeddings of multiple original embeddings, using invert_P().
    Writes cosine distance between inverted template and original embedding to textfile.

    **Inputs:**

    f : file handle
        Handle to opened file, in which the cosine distance between each inverted template and its original embedding will be written.

    overlap : int
        The amount of overlap between sets of embedding elements used to generate each PolyProtected element (0, 1, 2, 3, 4).

    Vs : 2D numpy array
        Contains all the input embeddings (1 row per embedding).

    Ps : 2D numpy array
        Contains the PolyProtected embeddings, each one stored in a separate array.

    Cs : 2D numpy array
        The coefficients used to generate the PolyProtected embeddings, each set stored in a separate array.

    Es : 2D numpy array
        The exponents used to generate the PolyProtected embeddings, each set stored in a separate array.

    num_guesses : scalar integer
        Number of guesses we should make for the embedding elements (V_1, ..., V_n), for the numerical solver.

    el_mins: 1D numpy array
        The minimum value of each embedding element's distribution.

    el_maxs: 1D numpy array
        The maximum value of each embedding element's distribution.

    el_probabilities : 2D numpy array
        The probabilities representing the histogram bins of each embedding element's distribution.

    precision : int
        The number of decimal places to which embedding elements should be rounded when calculating the possible values for the guesses.

    **Outputs:**

    (Textfile storing the cosine distances between the inverted PolyProtected templates and their original embeddings.)

    """

    for emb_ind in range(Vs.shape[0]): # for each input embedding
        print('-- Template %d of %d --' % (emb_ind + 1, Vs.shape[0]))
        # Calculate cosine distance between inverted template and its corresponding original embedding:
        solution_found, cosine_distance = invert_P(overlap, Vs[emb_ind, :], Ps[emb_ind], Cs[emb_ind], Es[emb_ind], num_guesses, el_mins, el_maxs, el_probabilities, precision) # NB: Different randomly initialised start values may be used for each embedding.
        # Write the calculated cosine distance to file (the distance will be "NaN" if no solution was found):
        f.write("%.8f\n" % (cosine_distance))

    return 0


def calc_threshold(extractor_model, fmr):
    """ Calculates the match threshold at the specified FMR, for the specified baseline face recognition system on the "dev" Mobio subset.

    **Inputs:**

    extractor_model : string
        Name of face recognition model (i.e., trained deep architecture) used to generate face embeddings.

    fmr : float
        Desired FMR value at which we wish to calculate the match threshold.

    **Outputs:**

    threshold : float
        The calculated threshold value.

    """

    # Define the path of the "dev" baseline score file:
    scores_path = "baselines/res/" + extractor_model + "_mobio/male/nonorm/scores-dev"

    # Read the score file and split the scores into "impostor" and "genuine" scores:
    [imp_scores, gen_scores] = split_four_column(scores_path)

    # Calculate the threshold at the specified FMR (which is the same as the FAR in this case):
    threshold = far_threshold(imp_scores, gen_scores, fmr)

    return threshold


def calc_irreversibility_rate(scores_filepath, threshold):
    """ Calculates the success rate of an attempt to reverse a PolyProtected template to recover the original embedding, based on the cosine distances calculated in
    the invert_multiple_Ps() function.  Success rate = solution count (i.e., number of non-NaN cosine distances) * match count (i.e., number of cosine distances >= provided match threshold).

    **Inputs:**

    scores_filepath : string
        The path of the textfile in which the irreversibility scores (i.e., cosine distances) are stored.

    threshold : float
        The cosine distance threshold established on the baseline "dev" set.

    **Outputs:**

    solution_rate : scalar float
        The proportion of input embeddings for which a solution was found.

    match_rate : scalar float
        The proportion of solutions that match their corresponding original embeddings, based on the provided cosine distance threshold.

    success_rate : scalar float
        The overall inversion success rate = solution_rate * match_rate

    """

    f = open(scores_filepath, "r")

    lines = f.readlines() # list of all lines from textfile
    lines_float_array = numpy.array([float(line) for line in lines]) # array of lines converted to floating point values
    # print(lines_float_array)

    solutions = lines_float_array[~numpy.isnan(lines_float_array)] # non-NaN values in lines_float_array
    # print(solutions)
    solution_cnt = len(solutions) # number of embeddings for which a suitable inverted template was found
    solution_rate = solution_cnt / len(lines_float_array) # proportion of embeddings for which a suitable inverted template was found
    # print("Solution rate: %.2f or %.2f%% (%d out of %d embeddings)" % (solution_rate, solution_rate * 100, solution_cnt, len(lines_float_array)))

    matches = solutions[solutions >= threshold] # inverted templates that match their corresponding original embeddings
    # print(matches)
    match_cnt = len(matches) # number of inverted templates that match their corresponding original embeddings
    match_rate = match_cnt / solution_cnt # proportion of inverted templates that match their corresponding original embeddings
    # print("Match rate: %.2f or %.2f%% (%d out of %d solutions)" % (match_rate, match_rate * 100, match_cnt, solution_cnt))

    success_rate = solution_rate * match_rate
    # print("Overall inversion success rate: %.2f or %.2f%%" % (success_rate, success_rate * 100))

    f.close()

    return solution_rate, match_rate, success_rate
