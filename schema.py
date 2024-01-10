import pandas as pd

# Schema for little tests
# Create an empty dataframe for short tests
cols = ['Test Number', 'Algorithm', 'Sample size', 'Total time']
for i in range(1, 11):
    cols.append(f'Protein {i}')
for i in range(1, 11):
    cols.append(f'rank {i}')
for i in range(1, 11):
    cols.append(f'percentile {i}')

scalability_test = pd.DataFrame(columns=cols)
print(scalability_test)

scalability_test.to_csv(r'Dissertation\data_files\tests_scale.csv', index=False)


#   Schema for accuracy tests
ac_cols = ['Test Number', 'Algorithm', 'Total time']
for i in range(1, 21):
    ac_cols.append(f'Protein {i}')
for i in range(1, 21):
    ac_cols.append(f'rank {i}')
for i in range(1, 21):
    ac_cols.append(f'percentile {i}')

accuracy_test = pd.DataFrame(columns=ac_cols)

print(accuracy_test)
accuracy_test.to_csv(r'Dissertation\data_files\tests_accuracy2.csv', index=False)


#   hyperparameter tuning schema
#   Create an empty DataFrame with the desired columns for testing against baseline
columns = ['Test Number', 'Algorithm', 'Hyperparameters', 'Start Time', 'End Time', 'Total Time',
           'rank 1', 'rank 2', 'rank 3', 'rank 4', 'rank 5', 'rank 6', 'rank 7', 'rank 8', 'rank 9', 'rank 10',
           'rank 11', 'rank 12', 'rank 13', 'rank 14', 'rank 15', 'rank 16', 'rank 17', 'rank 18', 'rank 19',
           'rank 20']

test_df = pd.DataFrame(columns=columns)

# Create an empty DataFrame for results against baseline
protein_columns = ['Test Number', 'Algorithm', 'Total Time']
for i in range(1, 21):
    protein_columns.append(f'Protein {i}')
for i in range(1, 21):
    protein_columns.append(f'rank {i}')
for i in range(1, 21):
    protein_columns.append(f'percentile {i}')

results_df = pd.DataFrame(columns=protein_columns)


test_df.to_csv(r'Dissertation\data_files\tests.csv', index=False)

print(test_df)

results_df = pd.DataFrame(columns=protein_columns)
results_df.to_csv(r'Dissertation\data_files\results.csv', index=False)