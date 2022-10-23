import os
import json
import pickle
import numpy as np
import pandas as pd
from astropy.io.votable import parse


def cluster_list(cluster_or_filename):
    """Function for creating a list of cluster names. If the input of this function is a file path,
    return a list of the lines in the file. If not, return a list with only the input string.

    Args:
        cluster_or_filename (str): Either the name of a cluster or the name of a file containing cluster names.

    Returns:
        cluster_names (str, list): List of cluster names.

    """
    if os.path.exists(cluster_or_filename):
        with open(cluster_or_filename, 'r') as f:
            cluster_names = f.read().splitlines()
        return cluster_names
    else:
        cluster_names = [cluster_or_filename]
        return cluster_names


def load_cluster_parameters(cluster_parameters_path, cluster_name):
    """Loads in the data of the cluster parameters and returns those of a single cluster.

    Args:
        cluster_parameters_path (str): Path to the (.csv) file where the cluster parameters are saved.
        cluster_name (str): Name of the cluster

    Returns:
        params (dict): Dictionary containing the parameters of a single cluster.

    """
    cluster_params = pd.read_csv(cluster_parameters_path, index_col=0).dropna()
    cluster_params = cluster_params.rename(columns={'Cluster': 'name', 'RA_ICRS': 'ra', 'DE_ICRS': 'dec',
                                                    'pmRA_': 'pmra', 'pmDE': 'pmdec', 'e_pmRA_': 'pmra_error',
                                                    'e_pmDE': 'pmdec_error', 'plx': 'parallax',
                                                    'e_plx': 'parallax_error', 'AgeNN': 'age', 'AVNN': 'a0',
                                                    'DistPc': 'dist'})
    if cluster_name in cluster_params['name'].values:
        params = cluster_params[cluster_params['name'] == cluster_name].to_dict(orient='list')
        params = {key: params[key][0] for key in params}
        return params
    else:
        return None


def load_cone(cone_path):
    """Loads in the data of the stars found in the cone search. Also performs a number of preprocessing steps.

    Args:
        cone_path (str): Path to the (vot.gz) file where the cone search data is saved.

    Returns:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.

    """
    cone = parse(cone_path)
    cone = cone.get_first_table().to_table(use_names_over_ids=True)
    cone = cone.to_pandas()
    return cone


def load_members(members_path, cluster_name):
    """Loads in the data of a membership list and selects the members of a single cluster.

    Args:
        members_path (str): Path to the (.csv) file where the member data is saved.
        cluster_name (str): Name of the cluster

    Returns:
        cluster_members (Dataframe): Dataframe containing the data of the cluster member sources.

    """
    all_members = pd.read_csv(members_path, index_col=0)
    cluster_members = all_members[all_members['cluster'] == cluster_name]
    cluster_members = cluster_members.reset_index(drop=True)[['source_id', 'PMemb']]
    return cluster_members


def load_isochrone(isochrone_path, age, dist, oldest_stage=7):
    """Loads in the data of the isochrones and selects the isochrone of a particular age.
    Also converts the isochrone's absolute G magnitude to apparent magnitude.

    Args:
        isochrone_path (str): Path to the (.dat) file where the isochrone data saved.
        age (float): Age of the cluster
        dist (float): Distance to the cluster
        oldest_stage (int): Oldest evolutionary stage to include in the isochrone (default is early ASG)

    Returns:
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.

    """
    isochrones = pd.read_csv(isochrone_path,
                             names=['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg',
                                    'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3',
                                    'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess',
                                    'Z', 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag'],
                             delim_whitespace=True,
                             comment='#')

    # Rename the relevant properties
    isochrones.rename(columns={'Gmag': 'phot_g_mean_mag'}, inplace=True)
    isochrones['bp_rp'] = isochrones['G_BPmag'] - isochrones['G_RPmag']

    # Select the isochrone with the age of the cluster
    ages = np.array(list(set(isochrones['logAge'].values)))
    closest_age = ages[np.argmin(np.abs(ages - age))]
    isochrone = isochrones[isochrones['logAge'] == closest_age]

    # Exclude data beyond the evolutionary stage of the early asymptotic giant branch
    isochrone = isochrone[isochrone['label'] <= oldest_stage].reset_index(drop=True)

    # Convert to apparent magnitude
    distance_modulus = 5 * np.log10(dist) - 5
    isochrone['phot_g_mean_mag'] += distance_modulus

    return isochrone[['bp_rp', 'phot_g_mean_mag']]


def save_sets(save_dir, train_members=None, candidates=None, non_members=None, comparison_members=None):
    """Saves the sets of sources in .csv files.

    Args:
        save_dir (str): Directory where the sets are saved
        train_members (Dataframe): Dataframe containing train member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources
        comparison_members (Dataframe): Dataframe containing comparison member sources

    """
    for subset, subset_name in zip([train_members, candidates, non_members, comparison_members],
                                   ['train_members', 'candidates', 'non_members', 'comparison_members']):
        if subset is not None:
            subset.to_csv(os.path.join(save_dir, subset_name + '.csv'))


def load_sets(save_dir):
    """Loads the sets of sources from .csv files.

    Args:
        save_dir (str): Directory where the sets are loaded from

    Returns:
        train_members (Dataframe): Dataframe containing train member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non-member sources
        comparison_members (Dataframe): Dataframe containing comparison member sources

    """
    files = ['train_members.csv', 'candidates.csv', 'non_members.csv', 'comparison_members.csv']
    sources = []
    for file in files:
        path = os.path.join(save_dir, file)
        if os.path.exists(path):
            sources.append(pd.read_csv(path, index_col=0))
        else:
            sources.append(None)
    train_members, candidates, non_members, comparison_members = tuple(sources)
    return train_members, candidates, non_members, comparison_members


def save_cluster(save_dir, cluster):
    """Saves the cluster parameters.

    Args:
        save_dir (str): Directory where the sets are saved
        cluster (Cluster): Cluster object

    """
    with open(os.path.join(save_dir, 'cluster'), 'wb') as cluster_file:
        pickle.dump(cluster, cluster_file)


def load_cluster(save_dir):
    """Loads the cluster parameters.

    Args:
        save_dir (str): Directory where the sets are saved

    Returns:
        cluster (Cluster): Cluster object

    """
    with open(os.path.join(save_dir, 'cluster'), 'rb') as cluster_file:
        cluster = pickle.load(cluster_file)
    return cluster


def save_hyper_parameters(model_save_dir, hyper_parameters):
    """Saves the hyperparameters of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model
            will be saved
        hyper_parameters (dict): Dictionary containing the data and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'hyper_parameters'), 'w') as f:
        json.dump(hyper_parameters, f)


def load_hyper_parameters(model_save_dir):
    """Loads the hyperparameters of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model
            are stored

    Returns:
        hyper_parameters (dict): Dictionary containing the data and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'hyper_parameters'), 'r') as f:
        hyper_parameters = json.load(f)
    return hyper_parameters


def save_metrics(model_save_dir, metrics_dict):
    """Saves the metrics of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model
            will be saved
        metrics_dict (dict): Dictionary containing and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'metrics'), 'w') as f:
        json.dump(metrics_dict, f)


def load_metrics(model_save_dir):
    """Loads the metrics of a certain model.

    Args:
        model_save_dir (str): Directory where the metrics corresponding to a specific model
            are stored

    Returns:
        metrics (dict): Dictionary containing and training metrics

    """
    with open(os.path.join(model_save_dir, 'metrics'), 'r') as f:
        metrics_dict = json.load(f)
    return metrics_dict
