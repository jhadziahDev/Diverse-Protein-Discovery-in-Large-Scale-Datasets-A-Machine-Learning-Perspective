import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import umap
from sklearn.decomposition import PCA
from tqdm import tqdm

path = r'C:\Users\jhadz\Documents\A_university_and_projects\Postgraduate\Dissertation\per_protein.h5'
metadata_path = r'C:\Users\jhadz\Documents\A_university_and_projects\Postgraduate\Dissertation\metadata.tsv.gz'


def load_embeddings_from_h5(h5_file_path):
    """ Extract embedding data from file and return data in dict format.

    :param h5_file_path: file path
    :return: dict of embedding data
    """
    # Get the total number of keys in the h5 file to set the progress bar's maximum value
    with h5py.File(h5_file_path, "r") as file:
        total_keys = len(file.keys())

    embeddings_dict = {}

    # Load the entire dataset from the h5 file with a progress bar
    with tqdm(total=total_keys, desc="Loading embeddings") as progress_bar:
        with h5py.File(h5_file_path, "r") as file:
            for key in file.keys():
                embeddings_dict[key] = file[key][:]
                progress_bar.update(1)  # Update the progress bar

    return embeddings_dict


def embedding_metadata(file_path):
    """ extract metadata on the protein embeddings

    :param file_path: file path to embedding metadata
    :return: pandas dataframe of embedding metadata
    """
    emb_data = pd.read_csv(file_path, sep='\t', compression='gzip')
    return emb_data


def nan_finder(emb):
    """ find NaNs in dataset, quantify total Nans, and return labels with NaNs

    :param emb: dict of embeddings

    """
    nan_count = 0
    for key, e in emb.items():
        nan_count += sum(np.isnan(val) for val in emb)

    print(f"There are {nan_count} NaN values in the embeddings.")

    labels_with_nans = {}

    for key, e in emb.items():
        nan_in_embedding = sum(np.isnan(val) for val in emb)
        if nan_in_embedding > 0:
            labels_with_nans[key] = nan_in_embedding

    for label, count in labels_with_nans.items():
        print(f"Label {label} has {count} NaN values.")


def plot_pca(labels, embedding_vals, meta_data, categories, EC=False, group=False):
    """plot the protein embeddings in 3d using PCA

    :param labels: list of protein labels
    :param embedding_vals: list of embedding values
    :param meta_data: dataframe containing the metadata and EC classifier for the proteins
    :param categories: which group of diversity the proteins are a part of
    :param EC: if true classify proteins by enzyme classification
    :param group: if true classify by diversity group
    """
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embedding_vals)

    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    if EC:
        #   get the metadata to group points
        label_to_ec = dict(zip(meta_data['Entry'], meta_data['EC number']))
        # print(label_to_ec)

        # Get EC numbers for each label
        ec_numbers = [label_to_ec.get(label, 'Unknown') for label in labels]

        ec_classes = ['NaN' if pd.isna(ec) else ec.split('.')[0] if isinstance(ec, str) else str(ec) for ec in
                      ec_numbers]
        unique_ecs = list(set(ec_classes))

        # Define colors for EC classes and a special color for NaN and Unknown values
        colors = px.colors.qualitative.Set1[:len(unique_ecs)]
        ec_to_color = {str(i): colors[i % len(colors)] for i in range(1, len(unique_ecs) + 1)}
        ec_to_color['NaN'] = 'gray'  # Set a special color for NaN values
        ec_to_color['Unknown'] = 'black'  # Set a special color for Unknown values

        # Create a dictionary to store coordinates for each EC class
        class_coords = {ec_class: {'x': [], 'y': [], 'z': []} for ec_class in unique_ecs}

        # Fill the dictionary with coordinates
        for i, ec_class in enumerate(ec_classes):
            class_coords[ec_class]['x'].append(x[i])
            class_coords[ec_class]['y'].append(y[i])
            class_coords[ec_class]['z'].append(z[i])

        # Create traces for each EC class
        traces = []
        for ec_class, coords in class_coords.items():
            trace = go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker=dict(
                    color=ec_to_color.get(ec_class, 'pink'),  # Use pink as the default color for unhandled EC classes
                    size=1,
                    opacity=0.5
                ),
                name=ec_class
            )
            traces.append(trace)

        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=60
            ),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.update_layout(title="Plot of protein embedding in 3D")
        fig.show()

    if group:
        #   group by diversity
        category_colors = {
            'top_500': 'red',
            'top_1000': 'blue',
            'top_2500': 'green',
            'top_5000': 'yellow',
            'top_10000': 'purple',
            'other': 'gray'
        }

        # Create a dictionary to store coordinates for each category
        class_coords = {category: {'x': [], 'y': [], 'z': []} for category in category_colors}

        # Fill the dictionary with coordinates
        for i, category in enumerate(categories):
            class_coords[category]['x'].append(x[i])
            class_coords[category]['y'].append(y[i])
            class_coords[category]['z'].append(z[i])

            # print("Length of categories:", len(categories))
            # print("Length of x:", len(x))

        # Create traces for each category
        traces = []
        for category, coords in class_coords.items():
            trace = go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker=dict(
                    color=category_colors.get(category, 'pink'),
                    size=1,
                    opacity=0.5
                ),
                name=category
            )
            traces.append(trace)

        layout = go.Layout(
            title="PCA of Protein Embeddings",
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=70
            ),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.update_layout(title="Plot of protein embeddings in 3D")
        fig.show()


def plot_umap(labels, embedding_vals, metadata, categories, EC=False, Group=False):
    """ PLot protein embeddings in 3D using UMAP

    :param labels: list of protein labels
    :param embedding_vals: list of embedding values
    :param metadata: dataframe containing the metadata and EC classifier for the proteins
    :param categories: which group of diversity the proteins are a part of
    :param EC: if true classify proteins by enzyme classification
    :param Group: if true classify by diversity group
    """
    # Use UMAP to reduce dimensions to 3D
    reducer = umap.UMAP(n_components=3, n_neighbors=500)
    reduced_embeddings = reducer.fit_transform(embedding_vals)

    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    z = reduced_embeddings[:, 2]

    if EC:
        # get the metadata to group points
        label_to_ec = dict(zip(metadata['Entry'], metadata['EC number']))
        # print(label_to_ec)

        # Get EC numbers for each label
        ec_numbers = [label_to_ec.get(label, 'Unknown') for label in labels]
        # print(ec_numbers)

        # extract the EC classes only for grouping
        ec_classes = ['NaN' if pd.isna(ec) else ec.split('.')[0] if isinstance(ec, str) else str(ec) for ec in
                      ec_numbers]

        # get the unique EC classess only
        unique_ecs = list(set(ec_classes))

        # Define colors for EC classes and a special color for NaN and Unknown values
        colors = px.colors.qualitative.Set1[:len(unique_ecs)]
        ec_to_color = {str(i): colors[i % len(colors)] for i in range(1, len(unique_ecs) + 1)}
        ec_to_color['NaN'] = 'gray'  # Set a special color for NaN values
        ec_to_color['Unknown'] = 'black'  # Set a special color for Unknown values

        # Create a dictionary to store coordinates for each EC class
        class_coords = {ec_class: {'x': [], 'y': [], 'z': []} for ec_class in unique_ecs}

        # Fill the dictionary with coordinates
        for i, ec_class in enumerate(ec_classes):
            class_coords[ec_class]['x'].append(x[i])
            class_coords[ec_class]['y'].append(y[i])
            class_coords[ec_class]['z'].append(z[i])

        # Create traces for each EC class
        traces = []
        for ec_class, coords in class_coords.items():
            trace = go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker=dict(
                    color=ec_to_color.get(ec_class, 'pink'),  # Use pink as the default color for unhandled EC classes
                    size=1,
                    opacity=0.5
                ),
                name=ec_class
            )
            traces.append(trace)

        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    if Group:
        #   group by diversity
        category_colors = {
            'top_500': 'red',
            'top_1000': 'blue',
            'top_2500': 'green',
            'top_5000': 'yellow',
            'top_10000': 'purple',
            'other': 'gray'
        }

        # Create a dictionary to store coordinates for each category
        class_coords = {category: {'x': [], 'y': [], 'z': []} for category in category_colors}

        # Fill the dictionary with coordinates
        for i, category in enumerate(categories):
            class_coords[category]['x'].append(x[i])
            class_coords[category]['y'].append(y[i])
            class_coords[category]['z'].append(z[i])

        # Create traces for each category
        traces = []
        for category, coords in class_coords.items():
            trace = go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker=dict(
                    color=category_colors.get(category, 'pink'),
                    size=1,
                    opacity=0.5
                ),
                name=category
            )
            traces.append(trace)

        layout = go.Layout(
            title="Umap of Protein Embeddings",
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    else:
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=4,
                opacity=0.6
            ),
            text=labels  # show protein labels when you hover over the points
        )

        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        fig = go.Figure(data=[trace], layout=layout)
        fig.show()


def categorise_based_on_index(idx):
    """Categorise protein based on its index/rank
    :return: categories based of the index of the protein embedding in the baseline brute force df
    """

    if idx < 500:
        return 'top_500'
    elif idx < 1000:
        return 'top_1000'
    elif idx < 2500:
        return 'top_2500'
    elif idx < 5000:
        return 'top_5000'
    elif idx < 10000:
        return 'top_10000'
    else:
        return 'other'


def main():
    #   get embeddings from file
    embeddings = load_embeddings_from_h5(path)

    # get embedding metadata
    meta_data = embedding_metadata(metadata_path)

    # explore data nans
    # nan_finder(embeddings)

    # Identify and remove keys with NaN values from the embeddings dictionary
    keys_to_remove = [key for key, embedding in embeddings.items() if np.isnan(embedding).any()]

    for key in keys_to_remove:
        del embeddings[key]

    # plot the most diverse proteins only
    baseline = pd.read_csv(
        r'C:\Users\jhadz\Documents\A_university_and_projects\Postgraduate\Learnt Modules\Python_projects\Dissertation\data_files\baseline_top_proteins_ordered.csv')
    top_500 = []
    top_1000 = []
    top_2500 = []
    top_5000 = []
    top_10000 = []

    for idx, row in baseline.iterrows():
        protein_label = row['Protein Label']

        if protein_label in embeddings:
            category = categorise_based_on_index(idx)

            if category == 'top_500':
                top_500.append(protein_label)
            elif category == 'top_1000':
                top_1000.append(protein_label)
            elif category == 'top_2500':
                top_2500.append(protein_label)
            elif category == 'top_5000':
                top_5000.append(protein_label)
            elif category == 'top_10000':
                top_10000.append(protein_label)

    categories = ['top_500'] * len(top_500) + ['top_1000'] * len(top_1000) + ['top_2500'] * len(top_2500) + [
        'top_5000'] * len(top_5000) + ['top_10000'] * len(top_10000)
    print(categories)

    most_diverse_labels = []
    most_diverse_values = []

    # Iterate over the first 10,000 items in the embeddings dictionary
    for i, (protein_label, embedding_value) in enumerate(embeddings.items()):
        if i >= 10000:
            break
        most_diverse_labels.append(protein_label)
        most_diverse_values.append(embedding_value)

    plot_pca(most_diverse_labels, most_diverse_values, meta_data, categories, False, True)
    plot_umap(most_diverse_labels, most_diverse_values, meta_data, categories, False, True)


    # plot all embeddings:
    protein_labels = list(embeddings.keys())
    embedding_values = list(embeddings.values())

    plot_pca(protein_labels, embedding_values, meta_data, categories, True)
    plot_umap(protein_labels, embedding_values, meta_data, categories,  True)

    # grouped by diversity
    plot_pca(protein_labels, embedding_values, meta_data, categories, False, True)
    plot_umap(protein_labels, embedding_values, meta_data, categories, False, True)

    #   Data analysis plots
    scale_df = pd.read_csv(r'Dissertation\data_files\test_scale_iters.csv')
    percentile_iters_df = pd.read_csv(r'Dissertation\data_files\percentiles_vs_iters.csv')

    #   plot scalability against iterations
    sns.set(style="whitegrid")

    # Create a FacetGrid to plot multiple plots in a grid
    g = sns.FacetGrid(scale_df, col="Algorithm", col_wrap=2, height=4, sharex=False, sharey=False)
    g.map(sns.lineplot, "Total Iterations", "Total Time (minutes)", marker="o")

    g.set_axis_labels("Total Iterations", "Time (minutes)")
    g.set_titles(col_template="{col_name}")
    g.tight_layout()

    plt.show()

    #   plot iterations vs performance
    # Melt the dataframe to have 'Total Iterations', 'variable' (percentile), and 'value' (count of proteins)
    df_melted = percentile_iters_df.melt(id_vars=['Algorithm', 'Total Iterations'],
                        value_vars=['Proteins above 95%', 'Proteins above 90%', 'Proteins above 80%'],
                        var_name='Percentile', value_name='Number of Proteins')

    sns.set(style="whitegrid")

    g = sns.relplot(x="Total Iterations", y="Number of Proteins", hue="Percentile", style="Percentile",
                    col="Algorithm", col_wrap=2, height=4, aspect=1.5, kind="line",
                    data=df_melted, facet_kws={'sharex': False, 'sharey': True})
    g.set_titles("{col_name}")

    plt.show()

    #   plot time vs performance

    # Plot using Seaborn
    df_melted2 = percentile_iters_df.melt(id_vars=['Algorithm', 'Total Time (minutes)'],
                        value_vars=['Proteins above 95%', 'Proteins above 90%', 'Proteins above 80%'],
                        var_name='Percentile', value_name='Number of Proteins')

    sns.set(style="whitegrid")

    g = sns.relplot(x="Total Time (minutes)", y="Number of Proteins", hue="Percentile", style="Percentile",
                    col="Algorithm", col_wrap=2, height=4, aspect=1.5, kind="line",
                    data=df_melted2, facet_kws={'sharex': False, 'sharey': True})
    g.set_titles("{col_name}")

    plt.show()

    # #   plot sample size vs time
    smp_scale_df = pd.read_csv(r'Dissertation\data_files\tests_scale.csv')

    # # Set the style of Seaborn
    sns.set(style="whitegrid")

    # Create a relplot for 'Total Time' against 'Sample size' for each algorithm
    g = sns.relplot(x="Total Time (minutes)", y="Sample size", hue="Algorithm", style="Algorithm",
                    col="Algorithm", col_wrap=2, height=4, aspect=1.5, kind="line",
                    data=smp_scale_df, facet_kws={'sharex': False, 'sharey': True})

    g.set_titles("{col_name}")
    g.set_axis_labels("Total Time", "Sample Size")

    plt.show()

    #   plot sample size vs performance

    # Melt the dataframe
    smp_scale_df_melted = smp_scale_df.melt(id_vars=['Sample size', 'Algorithm'],
                                            value_vars=['Proteins above 99.9%', 'Proteins above 99.5%',
                                                        'Proteins above 99%',
                                                        'Proteins above 95%', 'Proteins above 90%',
                                                        'Proteins above 80%'],
                                            var_name='Percentile', value_name='Value')

    sns.set(style="whitegrid")

    # Create a relplot for 'Sample size' against 'Percentile' for each algorithm using line plots
    g = sns.relplot(x="Sample size", y="Value", hue="Percentile",
                    col="Algorithm", col_wrap=2, height=4, aspect=1.5, kind="line",
                    markers=True, dashes=False, palette="tab10",
                    data=smp_scale_df_melted, facet_kws={'sharex': False, 'sharey': True})

    g.set_titles("{col_name}")
    g.set_axis_labels("Sample Size", "Number of Proteins")

    plt.show()

    #   plot range vs time
    sns.set(style="whitegrid")

    # Create a relplot for 'Total Time' against 'range' for each algorithm using line plots
    g = sns.relplot(x="Total Time", y="range", hue="Algorithm", col="Algorithm",
                    height=4, aspect=1, kind="line",
                    markers=True, dashes=False, data=scale_df,
                    facet_kws={'sharex': False, 'sharey': True})

    g.set_titles("{col_name}")
    g.set_axis_labels("Total Time", "Range")

    plt.show()

    #   plot range vs iterations
    sns.set(style="whitegrid")

    # Create a relplot for 'Total Time' against 'range' for each algorithm using line plots
    g = sns.relplot(x="Total Iterations", y="range", hue="Algorithm", col="Algorithm",
                    height=4, aspect=0.8, kind="line",
                    markers=True, dashes=False, data=scale_df,
                    facet_kws={'sharex': False, 'sharey': True})

    g.set_titles("{col_name}")
    g.set_axis_labels("Total Iterations", "Range")

    plt.show()


if __name__ == "__main__":
    main()
