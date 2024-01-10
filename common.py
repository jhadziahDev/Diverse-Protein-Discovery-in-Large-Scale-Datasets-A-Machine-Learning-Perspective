import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm


def find_maxsum_of_subset(subset):
    """ Calculate the brute force of a small sample subset of protein embeddings for comparison and validation

    :param subset: a dataframe where the first column is a protein label
    :return: dataframe containing the protein labels and their max sum embedding distances and a list of labels
    """

    labels = subset.iloc[:, 0]
    embeddings = subset.iloc[:, 1:]
    # to store results of max sum Euclidean distances
    maxsum_distances = []

    # For each embedding, calculate the sum of its Euclidean distances to all other embeddings
    for i, emb in embeddings.iterrows():
        distances = np.sqrt(((embeddings - emb) ** 2).sum(axis=1))
        maxsum_distance = distances.sum()
        maxsum_distances.append(maxsum_distance)

    # Create a new dataframe to store the results
    results = pd.DataFrame({
        'Protein Label': labels,
        'MaxSum Distance': maxsum_distances
    })

    # Sort the results by 'MaxSum Distance' in descending order
    sorted_results = results.sort_values(by=['MaxSum Distance'], ascending=False).reset_index(drop=True)

    return sorted_results, labels


def sample_embeddings(file_path, labels, keys=True, sample=False):
    """Extract either a subset or the full dataset from a h5 file

    :param file_path: path to embeddings data file
    :param labels: list of labels to extract
    :param sample: if True take a sample from the dataset else extract full dataset
    :param keys: if True return the data with protein labels
    :return: the extracted data in either a list with no labels or a dict with labels
    """

    with h5py.File(file_path, "r") as file:
        # Get the keys of the datasets in the H5 file
        dataset_keys = list(file.keys())

        if sample:
            # Convert labels to a set for faster lookup, ensures no duplicates
            labels_set = set(labels)
            # Get keys from the H5 file that are in labels
            wanted_keys = [k for k in dataset_keys if k in labels_set]

            # Iterate over the random keys and extract the corresponding embeddings
            if keys:
                # Initialize an empty dict to store the sampled embeddings
                sampled_embeddings = {}
                for key in wanted_keys:
                    embeddings = file[key][:]
                    sampled_embeddings[key] = embeddings
            else:
                # Initialize an empty list to store the sampled embeddings
                sampled_embeddings = []
                for key in wanted_keys:
                    embeddings = file[key][:]
                    sampled_embeddings.append(embeddings)

        else:
            if keys:
                # Initialize an empty dict to store the sampled embeddings
                sampled_embeddings = {}
                for key in dataset_keys:
                    embeddings = file[key][:]
                    sampled_embeddings[key] = embeddings
            else:
                # Initialize an empty list to store the sampled embeddings
                sampled_embeddings = []
                for key in dataset_keys:
                    embeddings = file[key][:]
                    sampled_embeddings.append(embeddings)

        return sampled_embeddings


def remove_nan_embeddings(data):
    """Remove any Nans in the given data

    :param data: dict of embeddings {protein:embedding}
    :return: dict, data cleaned of any nans
    """
    # Identify and remove keys with NaN values from the embeddings dictionary
    keys_to_remove = [key for key, embedding in data.items() if np.isnan(embedding).any()]

    for key in keys_to_remove:
        del data[key]

    return data


def load_embeddings_from_h5(h5_file_path):
    """Extract protein embeddings from h5 file with progress bar

    :param h5_file_path: full path to hdf file
    :return: dict, of the form {protein_label: array(embedding), ...}
    """
    embeddings = {}
    # Get the total number of keys in the h5 file to set the progress bar's maximum value
    with h5py.File(h5_file_path, "r") as file:
        total_keys = len(file.keys())
        # Load the entire dataset from the h5 file
        with tqdm(total=total_keys, desc="Loading embeddings") as progress_bar:
            # Load the entire dataset from the h5 file
            for key in file.keys():
                embeddings[key] = file[key][:]
                progress_bar.update(1)  # Update the progress bar
    return embeddings
