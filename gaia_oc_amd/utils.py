import numpy as np
import pandas as pd


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
