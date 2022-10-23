import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from gaia_oc_amd.io import load_sets


def projected_coordinates(ra, dec, cluster_ra, cluster_dec, cluster_dist):
    """Calculates the 2D projected coordinates, relative to the cluster center, from
    the right ascension and declination.

    Args:
        ra (float, array): Right ascension of sources
        dec (float, array): Declination of sources
        cluster_ra (float): Right ascension of the cluster
        cluster_dec (float): Declination of the cluster
        cluster_dist (float): Distance to the cluster

    Returns:
        x (float, array): The projected x coordinate
        y (float, array): The projected y coordinate

    """
    rad_per_deg = np.pi / 180

    ra = ra * rad_per_deg
    dec = dec * rad_per_deg
    ra_c = cluster_ra * rad_per_deg
    dec_c = cluster_dec * rad_per_deg

    d = cluster_dist

    x = d * np.sin(ra - ra_c) * np.cos(dec)
    y = d * (np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c))
    return x, y


def property_mean_and_std(data_dir, cluster_names, properties):
    """Calculates the mean and standard deviation of a list of properties. The mean and standard deviation
    are taken over the combined sources from the cone searches of the given clusters.

    Args:
        data_dir (str): Path to the directory containing the data
        cluster_names (str, list): Names of the clusters for which the sources are combined
        properties (str, list): List containing the property names

    Returns:
        mean (float, array): The mean of each property
        std (float, array): The standard deviation of each property

    """
    means = []
    stds = []

    # Store means and standard deviations of the training features for each cluster
    for cluster_name in tqdm(cluster_names, total=len(cluster_names), desc='Calculating means and stds...'):
        cluster_dir = os.path.join(data_dir, cluster_name)
        _, candidates, non_members, _ = load_sets(cluster_dir)

        all_sources = pd.concat((candidates, non_members))[properties].to_numpy()
        means.append(np.mean(all_sources, axis=0))
        stds.append(np.std(all_sources, axis=0))

    means = np.array(means)
    stds = np.array(stds)

    # Calculate the global feature means and standard deviations
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.mean(stds ** 2 + (means - mean) ** 2, axis=0))
    return mean, std


def add_columns(dataframes, functions, labels):
    """Adds a number of columns to a number dataframes.

    Args:
        dataframes (list, Dataframe): List of dataframes containing source data (e.g. members, non-members, etc.)
        functions (list, functions): List of functions which determine the column value.
        labels (list, str): List of labels for the added columns.
    """
    for dataframe in dataframes:
        if type(dataframe) == pd.DataFrame:
            for function, label in zip(functions, labels):
                result_type = None
                if type(label) == list:
                    result_type = 'expand'
                dataframe[label] = dataframe.apply(function, axis=1, result_type=result_type)
