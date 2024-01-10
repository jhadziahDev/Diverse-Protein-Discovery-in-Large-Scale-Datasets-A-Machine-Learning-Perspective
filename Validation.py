import pandas as pd
import h5py
import numpy as np
import time
from PE_algorithms import MaxSumTabuSearch
from PE_algorithms import TradMaxSumTabuSearchV1
from PE_algorithms import TradMaxSumTabuSearchV2
from PE_algorithms import MemeticGLS
from common import sample_embeddings
from common import remove_nan_embeddings
from common import find_maxsum_of_subset


def extract_embeddings(file_path, labels, keys=True, sample=False):
    ''' extract dataset from h5 file

    :param sample: if True take a sample from the dataset else extract full dataset
    :param labels: the labels of proteins for a set sample - to be used if sample is True
    :param keys: if true return the data with protein labels
    :param file_path: path to designated data file
    :return: the extarcted data in either a list with no labels or a dict with labels
    '''

    with h5py.File(file_path, "r") as file:
        # Get the keys of the datasets in the H5 file
        dataset_keys = list(file.keys())

        # Extract only a sample of embeddings
        if sample:
            # Convert labels to a set for faster lookup
            labels_set = set(labels)
            # Get keys from the H5 file that are in labels
            wanted_keys = [k for k in dataset_keys if k in labels_set]

            # Iterate over the random keys and extract the corresponding embeddings
            if keys:
                sampled_embeddings = {}
                for key in wanted_keys:
                    embeddings = file[key][:]
                    sampled_embeddings[key] = embeddings
            else:
                sampled_embeddings = []
                for key in wanted_keys:
                    embeddings = file[key][:]
                    sampled_embeddings.append(embeddings)

        #  Extract all embeddings
        else:
            if keys:
                sampled_embeddings = {}
                for key in dataset_keys:
                    embeddings = file[key][:]
                    sampled_embeddings[key] = embeddings
            else:
                sampled_embeddings = []
                for key in dataset_keys:
                    embeddings = file[key][:]
                    sampled_embeddings.append(embeddings)

        return sampled_embeddings


def result_match(val_df, protein_list):
    """Compare the output of an algorithm to the baseline validation

    :param val_df: dataframe containing the brute force calculations of all protein embeddings or of sample
    :param protein_list: the list of top proteins outputted from the tested algorithm
    :return: the index of where the protein was found in the val_df
    """
    found = False
    ranks = []
    for protein in protein_list:
        if protein in val_df['Protein Label'].values:
            found = True
            index = val_df[val_df['Protein Label'] == protein].index[0]  # Get the index of the protein in the dataframe
            print(f"Protein {protein} is in the validation set at index {index}.")
            ranks.append(index)

    if not found:
        print("None of the subset are in the validation set.")

    return ranks


def extract_csv(file_path):
    """ Extract csv file to pandas df

    :param file_path: file path of the csv file
    :return: pandas dataframe of csv data
    """
    return pd.read_csv(file_path)


def calculate_percentiles(ranks):
    """ calculate what percentile the "most diverse proteins" are in the brute force baseline

    :param ranks: list containing the indexes of the 'best proteins' from the brtue force baseline
    :return: list containing the percentiles that the 'best proteins' are in of the brute force baseline
    """
    total = 569507
    percentiles = []
    for rank in ranks:
        percentile = (rank / total) * 100
        percentiles.append(percentile)

    return percentiles


def run_and_compare(algorithm, baseline):
    """ run algorithms over a range of hyperparameters, compare results to baseline and write to datafile

    :param algorithm: the algorithm to be tested
    :param baseline: dataframe containing the brute force results for all protein embeddings
    :return: the list of diverse proteins, the ranks of the proteins, the percentiles of the proteins and total time
    to run the algorithm.
    """
    print(algorithm)
    print(str(algorithm))
    start_time = time.time()

    # Run the specific algorithm method and obtain results
    if isinstance(algorithm, MemeticGLS):
        protein_labels, *rest = algorithm.evolve_solution()
        print(protein_labels)
    else:
        if isinstance(algorithm, TradMaxSumTabuSearchV2):
            protein_labels, gl, ll = algorithm.run_tabu_search()
        else:
            protein_labels = algorithm.run_tabu_search()

        print('protein in run and compare', protein_labels)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate ranks and percentiles based on the baseline
    print(f'The validation results of {algorithm} against the baseline...')
    ranks = result_match(baseline, protein_labels)
    percentiles = calculate_percentiles(ranks)

    return protein_labels, ranks, percentiles, total_time, start_time, end_time


def main():
    # file paths
    path = r'Dissertation\data_files\Protein_emb.tsv'
    path_h5 = r'Dissertation\data_files\per_protein.h5'
    base_path = r'Dissertation\data_files\baseline_top_proteins_ordered.csv'
    test_path = r'Dissertation\data_files\tests.csv'
    results_path = r'Dissertation\data_files\results.csv'

    # get set sample of embeddings
    enb = pd.read_csv(path, sep='\t')
    print(enb)
    subset = enb.iloc[:101, :]
    print(subset)

    # get max sum of the set sample of embeddings
    subset_maxsum, labels = find_maxsum_of_subset(subset)
    print(subset_maxsum)

    # sample embeddings in correct format for tabu validation
    embeddings = extract_embeddings(path_h5, labels, True, True)

    # handle nans
    # Identify and remove keys with NaN values from the embeddings dictionary
    remove_nan_embeddings(embeddings)

    #   Begin Scale Tests
    scales = [101]
    test_path_sc = r'Dissertation\data_files\tests_scale.csv'
    scale_test = pd.read_csv(test_path_sc)

    #   Get set sample of embeddings of a certain size
    for scale in scales:
        subset = enb.iloc[:scale, :]
        print(subset)

        # get max sum of the set sample of embeddings
        subset_maxsum, labels = find_maxsum_of_subset(subset)
        print(subset_maxsum)

        # sample embeddings in correct format for testing
        embeddings = sample_embeddings(path_h5, labels, True, True)

        # Identify and remove keys with NaN values from the embeddings dictionary
        clean_emb = remove_nan_embeddings(embeddings)

        #   validate alternative tabu against set sample
        IATS = MaxSumTabuSearch(clean_emb, num_proteins=10, max_iterations=100)

        #   validate traditional tabu against set sample
        ITS1 = TradMaxSumTabuSearchV1(clean_emb, num_proteins=10, max_iterations=100)

        # validate traditional tabu mv2 against set sample
        ITS2 = TradMaxSumTabuSearchV2(clean_emb,
                                      num_proteins=10,
                                      max_iterations=50,
                                      local_iterations=100,
                                      local_sample_sizes=5)

        #   validate MemeticTABU against set sample
        mem_tabu = MemeticGLS(clean_emb,
                              dna_size=10,
                              max_epochs=1000,
                              local_iterations=500,
                              population_size=500,
                              retain_percent=0.05)

        algorithms = [IATS, ITS2, ITS1, mem_tabu]

        for algorithm in algorithms:
            testnum = 1
            try:
                # Run algorithm and compare against baseline
                protein_labels, ranks, percentiles, total, start_time, end_time = run_and_compare(algorithm,
                                                                                                  subset_maxsum)

                # Prepare new rows of data
                new_row = [testnum, algorithm,  scale, total]
                new_row.extend(protein_labels)
                new_row.extend(ranks)
                new_row.extend(percentiles)

                # Append the new rows to the DataFrames
                scale_test.loc[len(scale_test)] = new_row

                testnum += 1
            except Exception as e:
                print(f"Error running algorithm {algorithm}: {str(e)}")
                # scale_test.to_csv(test_path_sc, mode='a', header=False, index=False)

    try:
        # Save the dataframes to CSV files in append mode
        scale_test.to_csv(test_path_sc, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error writing to CSV files: {str(e)}")


    ##  Valadation against brute force baseline  ##
    baseline = extract_csv(base_path)
    results_df = extract_csv(results_path)
    all_embs = extract_embeddings(path_h5, labels, True)
    test_df = extract_csv(test_path)

    # Identify and remove keys with NaN values from the embeddings dictionary
    keys_to_remove = [key for key, embedding in all_embs.items() if np.isnan(embedding).any()]

    for key in keys_to_remove:
        del all_embs[key]

    # algorithms to be tested
    algorithms = [MaxSumTabuSearch, TradMaxSumTabuSearchV1, TradMaxSumTabuSearchV2, MemeticGLS]

    # hyperparameters to be tested
    hyperparameter_combinations = {
        MaxSumTabuSearch: [{'num_proteins': 20, 'max_iterations': 500, 'local_iterations': 500},
                           {'num_proteins': 20, 'max_iterations': 500, 'local_iterations': 300}],
        TradMaxSumTabuSearchV1: [{'num_proteins': 20, 'max_iterations': 600},
                                 {'num_proteins': 20, 'max_iterations': 250}],
        TradMaxSumTabuSearchV2: [
            {'num_proteins': 20, 'max_iterations': 500, 'local_iterations': 100, 'local_sample_sizes': 200},
            {'num_proteins': 20, 'max_iterations': 50, 'local_iterations': 1000, 'local_sample_sizes': 100}],
        MemeticGLS: [{'dna_size': 20, 'max_epochs': 1000, 'local_iterations': 500, 'population_size': 1000,
                      'retain_percent': 0.05},
                     {'dna_size': 20, 'max_epochs': 1000, 'local_iterations': 500, 'population_size': 5000,
                      'retain_percent': 0.05}]}

    # Loop through algorithms and their hyperparameter combinations
    for algorithm_class in algorithms:
        test_number = 3
        for hyperparams in hyperparameter_combinations[algorithm_class]:

            # Initialize algorithm with predefined parameters
            algorithm_instance = algorithm_class(all_embs, **hyperparams)

            try:
                # Run algorithm and compare against baseline
                protein_labels, ranks, percentiles, total, start_time, end_time = run_and_compare(algorithm_instance,
                                                                                                  baseline)
                # ranks analysis
                avg_rank = np.mean(ranks)
                low_rank = np.min(ranks)
                high_rank = np.max(ranks)
                median_rank = np.median(ranks)
                range_rank = high_rank - low_rank

                # time analysis
                tot_time_h = round(total / 3600, 3)
                tot_time_m = round(total / 60, 3)

                # percentile analysis
                avg_perc = np.mean(percentiles)
                value_at_perc95 = np.percentile(percentiles, 5)
                value_at_perc90 = np.percentile(percentiles, 10)
                value_at_perc80 = np.percentile(percentiles, 20)

                # Count the number of embeddings that are less than or equal to the values at the given percentiles
                perc95 = np.sum(np.array(percentiles) <= value_at_perc95)
                perc90 = np.sum(np.array(percentiles) <= value_at_perc90)
                perc80 = np.sum(np.array(percentiles) <= value_at_perc80)


                median_perc = np.median(percentiles)
                low_perc = np.min(percentiles)
                high_perc = np.max(percentiles)

                # Prepare new rows of data
                new_row = [test_number, algorithm_class.__name__, hyperparams, start_time, end_time, total]
                new_row_results = [test_number, algorithm_class.__name__, total]
                new_row_results.extend(protein_labels)
                new_row_results.extend(ranks)
                new_row.extend(ranks)
                # analysis_columns_test = [avg_rank, low_rank, high_rank, median_rank, range_rank, tot_time_h, tot_time_m]
                # analysis_columns_res = [avg_perc, perc95, perc90, perc80, avg_rank, low_rank, high_rank, median_rank,
                #                         range_rank, median_perc, low_perc, high_perc, tot_time_h, tot_time_m]
                # new_row.extend(analysis_columns_test)
                new_row_results.extend(percentiles)
                # new_row_results.extend(analysis_columns_res)

                # Append the new rows to the DataFrames
                test_df.loc[len(test_df)] = new_row
                results_df.loc[len(results_df)] = new_row_results

                test_number += 1
            except Exception as e:
                print(f"Error running algorithm {algorithm_class.__name__}: {str(e)}")

    try:
        # Save the dataframes to CSV files in append mode
        test_df.to_csv(test_path, mode='a', header=False, index=False)
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error writing to CSV files: {str(e)}")


    #   accuracy tests
    baseline = extract_csv(base_path)
    test_path_ac = r'Dissertation\data_files\tests_accuracy.csv'
    accuracy_test = pd.read_csv(test_path_ac)
    all_embs = extract_embeddings(path_h5, labels, True)

    # Identify and remove keys with NaN values from the embeddings dictionary
    keys_to_remove = [key for key, embedding in all_embs.items() if np.isnan(embedding).any()]

    for key in keys_to_remove:
        del all_embs[key]


    #   validate alternative tabu against baseline on best params
    IATS = MaxSumTabuSearch(all_embs, num_proteins=20, max_iterations=750)

    #   validate traditional tabu against baseline on best params
    ITS1 = TradMaxSumTabuSearchV1(all_embs, num_proteins=20, max_iterations=500)

    # validate traditional tabu mv2 against set sample
    ITS2 = TradMaxSumTabuSearchV2(all_embs,
                                  num_proteins=20,
                                  max_iterations=500,
                                  local_iterations=500,
                                  local_sample_sizes=100)

    #   validate MemeticTABU against set sample
    mem_tabu = MemeticGLS(all_embs,
                          dna_size=20,
                          max_epochs=5000,
                          local_iterations=500,
                          population_size=500,
                          retain_percent=0.05)

    algorithms = [ITS1, IATS]

    for i in range(3):
        for algorithm in algorithms:
            testnum = 1
            try:
                # Run algorithm and compare against baseline
                protein_labels, ranks, percentiles, total, start_time, end_time = run_and_compare(algorithm,
                                                                                                  baseline)

                # Prepare new rows of data
                new_row = [testnum, algorithm, total]
                new_row.extend(protein_labels)
                new_row.extend(ranks)
                new_row.extend(percentiles)

                # Append the new rows to the DataFrames
                accuracy_test.loc[len(accuracy_test)] = new_row

                testnum += 1
            except Exception as e:
                print(f"Error running algorithm {algorithm}: {str(e)}")
                # scale_test.to_csv(test_path_sc, mode='a', header=False, index=False)

    try:
        # Save the dataframes to CSV files in append mode
        accuracy_test.to_csv(test_path_ac, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error writing to CSV files: {str(e)}")


        #   Repeatability tests
        baseline = extract_csv(base_path)
        test_path_ac2 = r'Dissertation\data_files\tests_accuracy2.csv'
        accuracy_test = pd.read_csv(test_path_ac)
        all_embs = extract_embeddings(path_h5, labels, True)

        # Identify and remove keys with NaN values from the embeddings dictionary
        keys_to_remove = [key for key, embedding in all_embs.items() if np.isnan(embedding).any()]

        for key in keys_to_remove:
            del all_embs[key]

        #   validate alternative tabu against baseline on best params
        IATS = MaxSumTabuSearch(all_embs, num_proteins=20, max_iterations=750)

        #   validate traditional tabu against baseline on best params
        ITS1 = TradMaxSumTabuSearchV1(all_embs, num_proteins=20, max_iterations=500)

        # validate traditional tabu mv2 against set sample
        ITS2 = TradMaxSumTabuSearchV2(all_embs,
                                      num_proteins=20,
                                      max_iterations=500,
                                      local_iterations=500,
                                      local_sample_sizes=100)

        #   validate MemeticTABU against set sample
        mem_tabu = MemeticGLS(all_embs,
                              dna_size=20,
                              max_epochs=5000,
                              local_iterations=500,
                              population_size=500,
                              retain_percent=0.05)

        algorithms = [ITS2, mem_tabu]

        for i in range(3):
            for algorithm in algorithms:
                testnum = 1
                try:
                    # Run algorithm and compare against baseline
                    protein_labels, ranks, percentiles, total, start_time, end_time = run_and_compare(algorithm,
                                                                                                      baseline)

                    # Prepare new rows of data
                    new_row = [testnum, algorithm, total]
                    new_row.extend(protein_labels)
                    new_row.extend(ranks)
                    new_row.extend(percentiles)

                    # Append the new rows to the DataFrames
                    accuracy_test.loc[len(accuracy_test)] = new_row

                    testnum += 1
                except Exception as e:
                    print(f"Error running algorithm {algorithm}: {str(e)}")
                    # scale_test.to_csv(test_path_sc, mode='a', header=False, index=False)

        try:
            # Save the dataframes to CSV files in append mode
            accuracy_test.to_csv(test_path_ac2, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error writing to CSV files: {str(e)}")


if __name__ == "__main__":
    main()