""" Includes several functions used to process the raw face embeddings.

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




import h5py
import os
import numpy




def read_hdf5(filepath):
    """ Reads an embedding from the specified HDF5 file into a numpy array.

    **Inputs:**

    filepath : string
        Full path of the HDF5 file containing the embedding.

    **Outputs:**

    V : array of real numbers, both positive and negative
        The embedding.

    """

    f = h5py.File(filepath, 'r')
    V = f["array"][:]
    f.close()

    return V


def write_hdf5(filepath, P):
    """ Writes the PolyProtected embedding into an HDF5 file.

    **Inputs:**

    filepath : string
        Full path of the HDF5 file to which the PolyProtected embedding should be written.

    P : real number numpy array
        The PolyProtected template corresponding to the protected embedding.

    **Outputs:**

    HDF5 file containing the PolyProtected template.

    """

    f_new = h5py.File(filepath, 'w')
    f_new.create_dataset("array", data = P)
    f_new.close()


def read_reference_embeddings(dir_path):
    """ Reads all reference embeddings from their HDF5 files and stores them into a 2D numpy array.

    **Inputs:**

    dir_path : string
        Full path of the directory inside which the reference embedding HDF5 files are stored.

    **Outputs:**

    embeddings : 2D numpy array
        All the reference embeddings, each one stored in a different row.

    """

    files = os.listdir(dir_path)
    embeddings = []
    for file in files:
        embedding_filename = dir_path + file
        # print("embedding_filename: %s" % (embedding_filename))
        embedding_set = read_hdf5(embedding_filename) # there are 5 embeddings per person
        embedding = numpy.mean(embedding_set, axis = 0) # calculate average embedding for the current person
        embeddings.append(embedding)

    return numpy.array(embeddings)


def read_query_embeddings(dir_path):
    """ Reads all query embeddings from their HDF5 files and stores them into a 2D numpy array.

    **Inputs:**

    dir_path : string
        Full path of the directory inside which the query embedding HDF5 files are stored.

    **Outputs:**

    embeddings : 2D numpy array
        All the query embeddings, each one stored in a different row.

    """

    dirs = os.listdir(dir_path)
    embeddings = []
    for place in dirs:
        people = [p for p in os.listdir(os.path.join(dir_path + place))]
        for person in people:
            person_id = str(int(person[1 : len(person)])) # to ensure that, for example, '001' is the same as '1'
            sessions = os.listdir(os.path.join(dir_path + place + "/" + person))
            for session in sessions:
                person_files = os.listdir(os.path.join(dir_path + place + "/" + person + "/" + session))
                for p_file in person_files:
                    embedding_filename = os.path.join(dir_path + place + "/" + person + "/" + session + "/") + p_file
                    # print("embedding_filename: %s" % (embedding_filename))
                    embedding = read_hdf5(embedding_filename)
                    embeddings.append(embedding)

    return numpy.array(embeddings)


def calc_embedding_distributions(embeddings, precision):
    """ Calculates the probability distribution of each embedding element (i.e., dimension) separately.

    **Inputs:**

    embeddings : 2D numpy array
        A 2D numpy array of all embeddings that you wish to use to plot the distribution, where each row corresponds to a different embedding.

    precision : int
        Specifies the number of digits after the decimal point to which the embedding values should be rounded.

    **Outputs:**

    means : 1D numpy array (1 value per embedding element)
        The mean of each embedding element's distribution.

    mins : 1D numpy array (1 value per embedding element)
        The minimum of each embedding element's distribution.

    maxs : 1D numpy array (1 value per embedding element)
        The maximum of each embedding element's distribution.

    """

    num_elements = numpy.shape(embeddings)[1]

    means = numpy.zeros(num_elements)
    mins = numpy.zeros(num_elements)
    maxs = numpy.zeros(num_elements)
    probabilities = []

    step_size = 1.0 * (10 ** -precision) # histogram bin size

    # Loop through each element in all embeddings:
    for el_idx in range(0, num_elements):
        # Get all elements at a specific index from all embeddings:
        elements = embeddings[:, el_idx]
        # Round the element values to the specified precision (i.e., digits after the decimal point):
        elements_rounded = numpy.around(elements, decimals = precision)
        # Calculate the mean:
        means[el_idx] = numpy.mean(elements_rounded)
        # Calculate the minimum:
        minimum = min(elements_rounded)
        mins[el_idx] = minimum
        # Calculate the maximum:
        maximum = max(elements_rounded)
        maxs[el_idx] = maximum
        # Calculate the probability distribution in terms of a histogram:
        counts, bin_edges = numpy.histogram(elements_rounded, bins=numpy.arange(minimum, maximum + 2 * step_size, step_size))  # bin the element values (use 2 * step_size as last bin edge, so that max value is in its own (last) bin)
        probs = counts/float(len(elements_rounded))  # normalise the bin counts so that every bin value gives the probability of that bin
        probabilities.append(probs)

    return means, mins, maxs, probabilities
