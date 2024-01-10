Title: Diverse Protein Discovery in Large-Scale Datasets: A Machine Learning Perspective

## Description
This work project aims to find methods of identifying the most distant proteins and most diverse subsets of proteins
from large protein databases in a scalable and efficient way using a dataset of protein embeddings from SwissProt,
data mining techniques and metaheuristics. The project implements a brute force calculation of the MaxSum Euclidean
distance for each embedding as well as two variations of traditional tabu searches, an alternative tabu search and a
memetic tabu search.

The project contains the code for the algorithms, validation, testing, data processing, data cleaning, data analysis and
other supplementary code.

## Pre-requisites
The SwissProt Embedding and metadata datasets can be downloaded from here: https://www.uniprot.org/help/downloads
The embeddings used in this project are the reviewed by protein embeddings.

Ensure that the file locations of the downloaded embeddings are updated before running the project files.

# Packages
The following packages are needed:
Pandas
Numpy
DataClassess
Random
H5py
Matplotlib
Seaborn
Plotly
Scipy spatial distance
Sklearn for PCA
Tqdm
MPLcursors
Time
Umap
Cuda
Multiprocessing
Ast
Math
collections for defultdict
typing for list
warnings

## Contents
The python files are contained within the python package 'Dissertation'.

In this package is a file named 'Data files' which contains the following files:
baseline_top_proteins.csv - The output csv of the brute force calculation on the entire dataset containing protein
                            labels and their max sum values in decending order.
baseline_top_proteins_ordered.csv - The ordered version of the baseline_top_proteins.csv file.
metadata.tsv.gz - zip file containing the metadata for all embeddings for plots and visualisations.
per_protein.h5 - H5 file containing the protein embeddings.
percentiles_vs_iters.csv - csv file containing the results of tests for plotting.
protein_emb.tsv - tsv file version of the per_protein.h5 file used to easily access subsets of the data used during
                  code development, validation and scale tests.
results.csv - The results of the parameter test containing columns for the Test number, Algorithm name, 20 protein
              labels, 20 proteins ranks, 20 protein percentiles and columns for the summary statistics of the test.
results - Copy.csv - a copy of the results.csv file.
test_scale_iters.csv - csv file containing the results of the scale testing over set iterations
tests.csv - the results of the parameter testing. Contains columns for test number, algorithm name, total time (seconds)
            20 protein ranks, summary statistic columns and total time in hours and minutes.
tests_accuracy.csv - csv file containing the results of the accuracy/precision tests for the alternative tabu search and
                     traditional tabu search version 1 algorithms.
tests_accuracy2.csv - csv file of the accuracy/repeatability tests of the memetic tabu and traditional tabu version 2
                      algorithms.
test_scale.csv - csv file containing the results of the scale tests.
val_results.csv - a csv file containing the validation results.

This python package also contains the following python files:
Brute_force_MP.py - The multiprocessing version of the brute force calculation of the MaxSum pairwise euclidean distance.
Brute_force_MPGPU.py - The GPU accelerated multiprocessing version of the brute force MaxSum.
BruteForce.py - The original version of the brute force MaxSum.
common.py - python file containing some common functions used in the other python files.
data_analysis.py - python file where data cleaning and data analysis is conducted on the outputted result files of the
                   parameter, scale, precision, accuracy tests. The code adds missing headers, removes duplicate rows,
                   sorts rows, adds summary statistic columns to csv files, creates csv files for plotting and returns
                   some summary statistics on the input data. Testing must have been conducted beforehand.
Exploring_data.py - Python file containing the initial embedding data exploration. The code has the following
                    functionality: finds simple distances between two embeddings for cosine similarity, Euclidean
                    distance and Manhattan distance, use these same distance metrics to return the top 10 most distant
                    embeddings, initial creation of a brute force MaxSum, plot a subset of embeddings in 2D using PCA,
                    compare run times for brute force calculations over sample sizes.
PE_algorithms.py - Python file containing the metaheuristic algorithms. The algorithms are implemented as classes. The
                   The Classes are as follows: MaxSumTabuSearch is the alternative version of a tabu search,
                   TradMaxSumTabuSearchV1 is the first version of the traditional tabu search,
                   TradMaxSumTabuSearchV2 is the second version of the traditional tabu search,
                   ProteinSolution dataclass used to store the solutions for the memetic tabu search,
                   MemeticGLS the memetic tabu search with a genetic local search
plots_visualisations.py - Python file containing the plotting code. Plots the entire embeddings space in 3D using
                          dimension reduction techniques of PCA and UMAP. Plots the most distant embeddings in 3D using
                          PCA and UMAP. Classifies the embeddings by distance or Enzyme classification number. Also
                          plots the results of the tests.
schema.py - Python file containing code to make the csv files used to store the results of the tests.
to_tsv.py - Python file containing code to convert the embeddings from a H5 files to a tsv file.
Validation.py - Python file containing the code used to test the algorithms. Contains functions to extract the
                embeddings with the option to take a sample, to identify the psuedo rank of the protein embeddings in the
                output subset of the tested algorithm with the index of the brute force baseline, and code to test the
                algorithms. Contains tests for parameter tuning, performance over scale, repeatability and precision.


## Usage
In the current state the plotting python file plots_visualisations.py can be run without changes.
Additionally, no changes are needed to run Exploring_data.py.
To avoid errors when running the validation.py, re-run schema with a new file name and update the relevant parts of
the code referring to the file path. Errors may occur because the current result files have had the additional data
analysis columns added from data_analysis.py and will result in a dataframe mismatched column error.

alternatively:

Firstly remove current result files.
Run schema.py to create new result files to store test results
Run validation.py to test algorithms
Run data_analysis.py
Run plots_visualisations.py
