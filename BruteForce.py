import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm


def find_maxsum_of_subset(subset):
    """
    :param subset: a dataframe where the first column is a label
    :return: dataframe containing the protein labels and their max sum embedding distances and a list of labels
    """
    labels = subset.iloc[:, 0]
    embeddings = subset.iloc[:, 1:]

    # Initialize an empty list to store results of max sum euclidean distances
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
    sorted_results = results.sort_values(by='MaxSum Distance', ascending=False).reset_index(drop=True)

    return sorted_results, labels


def sample_embeddings(file_path, labels, keys=True, sample=False):
    ''' extract dataset from h5 file

    :param sample: if True take a sample from the dataset else extract full dataset
    :param labels:
    :param keys: if true return the data with protein labels
    :param file_path: path to designated data file
    :return: the extarcted data in either a list with no labels or a dict with labels
    '''

    with h5py.File(file_path, "r") as file:
        # Get the keys of the datasets in the H5 file
        dataset_keys = list(file.keys())

        if sample:
            # Convert labels to a set for faster lookup
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


def brute_force_max_sum(sample_emb, num_proteins=10, full=False):
    """ Calculate the brute force max sum of euclidean distances between embeddings

    :param sample_emb: dictionary containing an embedding label and embedding values
    :param num_proteins: how many diverse embeddings to select for the final subset
    :param full: whether the full dataset or a sample of the dataset is being used, if full save most diverse to a csv
    :return: The list of embedding labels of the most diverse embeddings
    """
    # Convert the dictionary to a list of tuples
    sample_tuples = list(sample_emb.items())

    # Initialize a list to store the diversity scores for each protein
    diversity_scores = []

    # Compute the total number of iterations
    total_iterations = len(sample_tuples) * (len(sample_tuples) - 1)

    # Initialize the progress bar
    progress_bar = tqdm(total=total_iterations, desc="Calculating diversity", unit="iteration")

    # Iterate over each protein embedding and label
    for label, embedding in sample_tuples:
        # Initialize the diversity score for the current protein
        diversity_score = 0

        # Calculate the sum of distances between the current protein and all other proteins
        for other_label, other_embedding in sample_tuples:
            if label != other_label:
                distance = np.linalg.norm(embedding - other_embedding)  # euclidean distance
                diversity_score += distance

            # Update the progress bar
            progress_bar.update()

        # Append the diversity score and protein label to the list
        diversity_scores.append((diversity_score, label))

    # Close the progress bar
    progress_bar.close()

    # Sort the diversity scores in descending order
    diversity_scores.sort(reverse=True)

    # If 'full' is True, save the results to a CSV file
    if full:
        df = pd.DataFrame(diversity_scores, columns=['MaxSum Value', 'Protein Label'])
        df.to_csv('top_proteins.csv', index=False)

    # Extract the top 'num_proteins' most diverse proteins
    top_proteins = [protein_label for _, protein_label in diversity_scores[:num_proteins]]

    return top_proteins


def compute_diversity_for_chunk(chunk, all_samples):
    """ Calculate the Max sum pairwise euclidean distance (diversity) for embeddings in chunk

    :param chunk: Embeddings to calculate max sum
    :param all_samples: All other embeddings in the dataset
    :return: List of diversity scores for the embeddings in chunk
    """
    diversity_scores_chunk = []
    for label, embedding in chunk:
        diversity_score = 0
        for other_label, other_embedding in all_samples:
            if label != other_label:
                distance = np.linalg.norm(embedding - other_embedding)  # euclidean distance
                diversity_score += distance
        diversity_scores_chunk.append((diversity_score, label))

    # indicating this chunk is done
    print(f"Chunk completed: {chunk[0][0]} to {chunk[-1][0]}")
    return diversity_scores_chunk


def brute_force_max_sum_parallel(sample_emb, num_proteins=10, full=False):
    """ Use parallel processing to Calculate the brute force max sum of euclidean distances between embeddings

        :param sample_emb: dictionary containing an embedding label and embedding values
        :param num_proteins: how many diverse embeddings to select for the final subset
        :param full: whether the full dataset or a sample of the dataset is being used, if full save most diverse to csv
        :return: The list of embedding labels of the most diverse embeddings
        """
    # Convert the dictionary to a list of tuples
    vals = list(sample_emb.values())
    sample_tuples = list(sample_emb.items())
    print(vals[0].shape)


    # Create a pool of worker processes
    num_processes = mp.cpu_count()
    pool = Pool(processes=num_processes)

    # Split the samples into chunks
    chunk_size = len(sample_tuples) // num_processes
    chunks = [sample_tuples[i:i + chunk_size] for i in range(0, len(sample_tuples), chunk_size)]

    # Use the pool to compute the diversity scores in parallel
    results = pool.starmap(compute_diversity_for_chunk, [(chunk, sample_tuples) for chunk in chunks])

    pool.close()
    pool.join()

    # Flatten the results list
    diversity_scores = [item for sublist in results for item in sublist]

    # Sort the diversity scores in descending order
    diversity_scores.sort(reverse=True)

    # If 'full' is True, save the results to a CSV file
    if full:
        df = pd.DataFrame(diversity_scores, columns=['MaxSum Value', 'Protein Label'])
        df.to_csv('top_proteins.csv', index=False)

    # Extract the top 'num_proteins' most diverse proteins
    top_proteins = [protein_label for _, protein_label in diversity_scores[:num_proteins]]

    return top_proteins

def main():
    path = r'Dissertation\Protein_emb.tsv'
    path_h5 = r'Dissertation\per_protein.h5'

    # get set sample of embeddings
    enb = pd.read_csv(path, sep='\t')
    print(enb)

    subset = enb.iloc[:101, :]
    print(subset)

    # get max sum of the set sample of embeddings
    subset_maxsum, labels = find_maxsum_of_subset(subset)
    print(subset_maxsum)

    # validate brute force function
    embeddings = sample_embeddings(path_h5, labels, True, True)

    top_proteins = brute_force_max_sum(embeddings, 10)
    print(top_proteins)

    # calculate the brute force MaxSum of euclidean distances for entire dataset
    full_embeddings = sample_embeddings(path_h5, labels, True)
    # num_keys = len(full_embeddings)
    # print(num_keys)

    top1000_proteins = brute_force_max_sum(full_embeddings, 1000, True)
    print(top1000_proteins)




if __name__ == "__main__":
    main()