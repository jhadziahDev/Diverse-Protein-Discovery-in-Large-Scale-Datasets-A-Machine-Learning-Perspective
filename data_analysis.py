import pandas as pd
import ast


def add_header(file, header):
    """Add header if header is missing from dataframe/csv file

    :param file: CSV file
    :param header: the header row
    """
    # Read the existing CSV file without a header
    df = pd.read_csv(file, header=None)

    # Create a new DataFrame with the header row
    new_df = pd.DataFrame([header])

    # Concatenate the new DataFrame with the existing DataFrame
    final_df = pd.concat([new_df, df], ignore_index=True)

    # Save the final DataFrame to the CSV file
    final_df.to_csv(file, index=False, header=False)


def clean_dataframes(file):
    """Make sure there is no duplicate rows in the dataframe

    :param file: Path to desired csv file
    """

    df = pd.read_csv(file)
    print(df)

    # drop duplicate rows ignoring the Test number column
    unique_df = df.loc[df.drop('Test Number', axis=1).drop_duplicates().index]

    # save to file
    unique_df.to_csv(file, index=False)


def recalculate_percentiles(row):
    """Recalculate the rank percentile for protein embeddings if there are nans from failed test runs

    :param row: rows with missing percentiles
    :return: the rows of re-calculated percentiles
    """
    total = 569507
    ranks = row.loc['rank 1':'rank 20'].values
    # calculate percentage
    percentiles = [(rank / total) * 100 for rank in ranks]
    return percentiles


def analyse_ranks(file_name):
    """Analyse the ranks columns and add an average rank, median, min, max, standard deviation and range column to the
    dataframe if there is not already a matching column

    :param file_name: The file path to the csv file
    :return: the dataframe with or without updates.
    """
    df = pd.read_csv(file_name)

    # function to apply the columns on either 10 or 20 ranks
    def apply_operation(operation, column_name, operation_name):
        if column_name not in df.columns:
            if 'rank 20' in df.columns:
                df[column_name] = operation(df.loc[:, 'rank 1':'rank 20'], axis=1)
            else:
                df[column_name] = operation(df.loc[:, 'rank 1':'rank 10'], axis=1)

    # Apply columns
    apply_operation(pd.DataFrame.mean, 'average rank', 'mean')
    apply_operation(pd.DataFrame.median, 'Median rank', 'median')
    apply_operation(pd.DataFrame.min, 'lowest rank', 'min')
    apply_operation(pd.DataFrame.max, 'highest rank', 'max')
    apply_operation(pd.DataFrame.std, 'standard deviation', 'std')

    # Adding a column for range
    if 'range' not in df.columns:
        df['range'] = df['highest rank'] - df['lowest rank']

    # Write the updated DataFrame back to the CSV
    df.to_csv(file_name, index=False)

    return df


def percentile_analysis(file_name):
    """Analyse the percentile rows and add columns with the top 99.9%, 99.5%, 99%, 95%, 90% and 80% percentile,
    highest and lowest percentile, median and mean percentiles. Write the new dataframe to CSV.

    :param file_name: The file path to the csv file
    :return: the dataframe with or without updates.
    """
    df = pd.read_csv(file_name)

    # Define percentiles.
    if 'percentile 20' in df.columns:
        percentiles = df.loc[:, 'percentile 1':'percentile 20']
    else:
        percentiles = df.loc[:, 'percentile 1':'percentile 10']

    # If 'average percentile' column doesn't exist, then compute and append it
    if 'Average percentile' not in df.columns:
        # Check if the dataframe has up to 'rank 20' or only up to 'rank 10'
        if 'percentile 20' in df.columns:
            df['Average percentile'] = df.loc[:, 'percentile 1':'percentile 20'].mean(axis=1)
        else:
            df['Average percentile'] = df.loc[:, 'percentile 1':'percentile 10'].mean(axis=1)

    #  If 'median percentile' column doesn't exist, then compute and append it
    if 'Median percentile' not in df.columns:
        # Check if the dataframe has up to 'percentile 20' or only up to 'percentile 10'
        if 'rank 20' in df.columns:
            df['Median percentile'] = df.loc[:, 'percentile 1':'percentile 20'].median(axis=1)
        else:
            df['Median percentile'] = df.loc[:, 'percentile 1':'percentile 10'].median(axis=1)

    analysis_cols = ['Proteins above 99.9%', 'Proteins above 99.5%', 'Proteins above 99%',
                     'Proteins above 95%', 'Proteins above 90%', 'Proteins above 80%']

    for col in analysis_cols:
        if col not in df.columns:
            # Calculate how many protein embeddings are in the top percentiles and add to new columns
            df['Proteins above 99.9%'] = percentiles.apply(lambda row: (row <= 0.01).sum(), axis=1)
            df['Proteins above 99.5%'] = percentiles.apply(lambda row: (row <= 0.5).sum(), axis=1)
            df['Proteins above 99%'] = percentiles.apply(lambda row: (row <= 1).sum(), axis=1)
            df['Proteins above 95%'] = percentiles.apply(lambda row: (row <= 5).sum(), axis=1)
            df['Proteins above 90%'] = percentiles.apply(lambda row: (row <= 10).sum(), axis=1)
            df['Proteins above 80%'] = percentiles.apply(lambda row: (row <= 20).sum(), axis=1)

    # Add lowest and highest percentile columns
    if 'Lowest Percentile' not in df.columns:
        df['Lowest Percentile'] = percentiles.min(axis=1).apply(lambda x: f"{x:.2f}%")

    if 'Highest Percentile' not in df.columns:
        df['Highest Percentile'] = percentiles.max(axis=1).apply(lambda x: f"{x:.2f}%")

    # Write the updated DataFrame back to the CSV
    df.to_csv(file_name, index=False)

    return df


def time_analysis(file_name):
    """Convert the 'Total Time' column from seconds to hours and minutes.

    :param file_name: The file path to the csv file
    :return: the dataframe with or without updates.
    """

    df = pd.read_csv(file_name)

    # Convert 'Total Time' from seconds to hours and add to new column
    df['Total Time (hours)'] = df['Total Time'] / 3600

    # Convert 'Total Time' from seconds to minutes and add to new column
    df['Total Time (minutes)'] = df['Total Time'] / 60

    #  round the new columns to 3 decimal places:
    df['Total Time (hours)'] = df['Total Time (hours)'].round(3)
    df['Total Time (minutes)'] = df['Total Time (minutes)'].round(3)

    # Write the updated DataFrame back to the CSV
    df.to_csv(file_name, index=False)

    return df


def sort_dataframe(file_name):
    """Sort the dataframe by Algorithm and Test Number columns.

    :param file_name: The file path to the csv file
    :return: the sorted dataframe
    """
    df = pd.read_csv(file_name)

    # Sort by 'Algorithm' and 'Test Number'
    df_sorted = df.sort_values(by=['Algorithm', 'Test Number'])

    # Save the sorted DataFrame back to the CSV
    df_sorted.to_csv(file_name, index=False)

    return df_sorted


def get_iterations(hyperparam_str):
    """Extract iterations from the hyperparameter column of a dataframe and calculate the total iterations

    :param hyperparam_str: Column with the hyperparameter dictionaries
    :return: list or iterations
    """
    # Convert the string representation of dictionary to actual dictionary
    hyperparams = ast.literal_eval(hyperparam_str)
    if 'max_iterations' in hyperparams:
        if 'local_iterations' in hyperparams:
            return hyperparams['max_iterations'] * hyperparams['local_iterations']
        else:
            return hyperparams['max_iterations']
    if 'max_epochs' in hyperparams:
        if 'local_iterations' in hyperparams:
            return hyperparams['max_epochs'] * hyperparams['local_iterations']
        else:
            return hyperparams['max_epochs']
    else:
        return None


def main():
    file_path = r'Dissertation\25_mostdiverse.tsv.gz'
    test_path = r'Dissertation\data_files\tests.csv'
    results_path = r'Dissertation\data_files\results.csv'
    scale_Test = r'Dissertation\data_files\tests_scale.csv'
    scale_Test_copy = r'Dissertation\data_files\tests_scale - Copy.csv'
    val = r'Dissertation\data_files\val_results.csv'
    accuracy_test = r'Dissertation\data_files\tests_accuracy.csv'
    acc_test2 = r'Dissertation\data_files\tests_accuracy2.csv'

    most_diverse_df = pd.read_csv(file_path, sep='\t', compression='gzip')
    print(most_diverse_df.columns)
    print(most_diverse_df.Organism)
    print(most_diverse_df['Function [CC]'])

    # Define the column names
    columns = ['Test Number', 'Algorithm', 'Hyperparameters', 'Start Time', 'End Time', 'Total Time',
               'rank 1', 'rank 2', 'rank 3', 'rank 4', 'rank 5', 'rank 6', 'rank 7', 'rank 8', 'rank 9', 'rank 10',
               'rank 11', 'rank 12', 'rank 13', 'rank 14', 'rank 15', 'rank 16', 'rank 17', 'rank 18', 'rank 19',
               'rank 20']

    #   Add missing headers
    add_header(test_path, columns)
    add_header(results_path, columns)

    #   Clean dataframes removing duplicates.
    clean_dataframes(test_path)
    clean_dataframes(results_path)
    clean_dataframes(scale_Test)
    clean_dataframes(val)
    clean_dataframes(acc_test2)

    #   Analyse data of the validation results
    val1 = analyse_ranks(val)
    val2 = percentile_analysis(val)
    val3 =time_analysis(val)

    #   Analyse the data of the parameter test results
    # test dataframe
    tdf = pd.read_csv(test_path)

    tdf_with_av = analyse_ranks(test_path)

    tdf_sorted = sort_dataframe(test_path)

    tdf_time_an = time_analysis(test_path)

    # results dataframe
    rdf = pd.read_csv(results_path)

    # Check for rows with missing percentiles and recalculate
    missing_percentile_rows = rdf[rdf.loc[:, 'percentile 1':'percentile 20'].isnull().any(axis=1)]
    for idx, row in missing_percentile_rows.iterrows():
        rdf.loc[idx, 'percentile 1':'percentile 20'] = recalculate_percentiles(row)

    # Overwrite the original CSV with the updated DataFrame
    rdf.to_csv(results_path, index=False)

    rdf_average_anal = analyse_ranks(results_path)

    rdf_percentile_anal = percentile_analysis(results_path)

    rdf_time_anal = time_analysis(results_path)

    rdf_sorted = sort_dataframe(results_path)


    #   Analyse the results of the Scale testing
    tsdf = pd.read_csv(scale_Test)
    tsdfc = pd.read_csv(scale_Test_copy)

    copy_ranks = analyse_ranks(scale_Test)
    copy_percentiles = percentile_analysis(scale_Test)
    copy_time = time_analysis(scale_Test)
    copy_sorted = sort_dataframe(scale_Test)

    #   Analyse the results of the Accuracy test

    ac_ranks = analyse_ranks(accuracy_test)
    ac_percentiles = percentile_analysis(accuracy_test)
    ac_time = time_analysis(accuracy_test)
    sorted_ac = sort_dataframe(accuracy_test)

    #   Analyse the results of the repeatability test
    ac_ranks = analyse_ranks(acc_test2)
    ac_percentiles = percentile_analysis(acc_test2)
    ac_time = time_analysis(acc_test2)
    sorted_ac = sort_dataframe(acc_test2)

    #   Output results

    # Extract the lowest value in the Range
    rdf['Lowest Range'] = rdf['range'].apply(lambda x: int(x))

    # Group by 'Algorithm Name' and get the index of the row with the minimum 'Lowest Range'
    idx = rdf.groupby('Algorithm')['Lowest Range'].idxmin()

    # Use the index to extract the rows with the minimum 'Lowest Range' for each algorithm
    summary = rdf.loc[idx, ['Test Number', 'Algorithm', 'Lowest Range']].reset_index(drop=True)

    print(summary)

    # Extract the highest value in the Range
    rdf['Highest Range'] = rdf['range'].apply(lambda x: int(x))

    # Group by 'Algorithm Name' and get the index of the row with the 'Highest Range'
    idx = rdf.groupby('Algorithm')['Highest Range'].idxmax()

    # Use the index to extract the rows with the 'Highest Range' for each algorithm
    summary = rdf.loc[idx, ['Test Number', 'Algorithm', 'Highest Range']].reset_index(drop=True)

    print(summary)

    # Extract the highest value in the
    rdf['Highest Range'] = rdf['range'].apply(lambda x: int(x))

    # Group by 'Algorithm Name' and get the index of the row with the 'Highest Range'
    idx = rdf.groupby('Algorithm')['Highest Range'].idxmax()

    # Use the index to extract the rows with the 'Highest Range' for each algorithm
    summary = rdf.loc[idx, ['Test Number', 'Algorithm', 'Highest Range']].reset_index(drop=True)

    print(summary)


    #   prepare the dataframe for plotting scalability over iterations
    # Extract required columns and create new columns for hours and minutes
    scale_df = tdf[['Test Number', 'Algorithm', 'Total Time']].copy()
    scale_df['Total Time (hours)'] = tdf['Total Time'] / 3600
    scale_df['Total Time (minutes)'] = (tdf['Total Time'] % 3600) / 60

    scale_df['Total Iterations'] = tdf['Hyperparameters'].apply(get_iterations)
    print(scale_df)
    scale_df.to_csv(r'Dissertation\data_files\test_scale_iters.csv', index=False)

    scdf = pd.read_csv(r'Dissertation\data_files\test_scale_iters.csv')
    scale_df_sorted = sort_dataframe(scdf)


if __name__ == "__main__":
    main()


