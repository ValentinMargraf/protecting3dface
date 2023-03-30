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
<http://www.gnu.org/licenses/>.

modified by Valentin Margraf

"""




import numpy
import numpy as np
import math
from scipy.optimize import least_squares
from scipy.spatial.distance import cosine
#from pwkit.lmmin import Problem, ResidualProblem

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
    jacobi = np.zeros((2*height,2*height))

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
    jacobi=jacobi[:2*n,:2*n]
    return jacobi

def system(v, *data):
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
    f = numpy.zeros(n)
    f=np.array(f,dtype=complex)
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


def system_complex(v, *data):

    fx = system(v[:128] + 1j*v[-128:], *data)
    return np.concatenate((fx.real, fx.imag))


def invert_P_complex(overlap, V, P, C, E, num_guesses, el_mins, el_maxs, el_probabilities, precision):

    n=128
    #print(n)

    # Define the step size for the guess values:
    step_size = 1.0 * (10 ** -precision)
    guesses = numpy.zeros([n, num_guesses]) # 1 row per embedding dimension, 1 column per guess
    # print('Generating %d guesses ...' % (num_guesses))
    for el_idx in range(n): # For each embedding element (dimension)
        # Get the possible values from its corresponding histogram parameters
        min_value = el_mins[el_idx]
        max_value = el_maxs[el_idx]
        #possible_values = numpy.around(numpy.linspace(min_value, max_value + step_size, num_guesses), decimals = precision)
        possible_values = numpy.around(numpy.arange(min_value, max_value + step_size, step_size), decimals = precision)
        probabilities = el_probabilities[el_idx]
        crnt_guesses = numpy.random.choice(possible_values, size = [1, num_guesses], replace = True, p = probabilities) # no check for repeated set of guesses,
        guesses[el_idx, :] = crnt_guesses


    # For each set of guesses, try to solve for v_1 ... v_n (i.e., try to recover the n elements of the input embedding, V):
    solution_found = 0
    cosine_distance = float('NaN')
    # print('Solving ...')
    iter_=0
    for guess_idx in range(num_guesses): # for each of our num_guesses systems of n equations
        guesses_set_real = guesses[:, guess_idx] # current set of n guesses
        data = (overlap, C, E, P, n)
        guesses_set_imag=np.zeros(128)


        guesses_set = np.concatenate((guesses_set_real,guesses_set_imag))

        sol = least_squares(system_complex, guesses_set,args = data, method = 'lm',verbose=0)

           
        iter_+=1

        if sol.success == 1:
            #print('Solution found!')
            solution_found = 1
            sol_=sol.x[:128] + sol.x[-128:]*1j
            
            cosine_distance=cosine(sol.x[:128],V)
            break # no need to look for other solutions (i.e., as far as attacker is concerned, solution has been found)
        elif sol.success != 1:
            continue

    return sol, cosine_distance



