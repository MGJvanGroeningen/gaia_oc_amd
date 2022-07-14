import os
import json
import numpy as np
import pandas as pd
from astropy.io.votable import parse

from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.utils import extinction_correction, unsplit_sky_positions, add_columns, \
    phot_g_mean_mag_error_function, bp_rp_error_function


def cluster_list(cluster_or_filename, data_path):
    """Returns a list of cluster names based on a single cluster name or a file containing cluster names.

    Args:
        cluster_or_filename (str): Either the name of a cluster or the name of a file containing cluster names.
        data_path (str): Path to the data directory

    Returns:
        cluster_names (str, list): List of cluster names.

    """
    path = find_path(cluster_or_filename, data_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            cluster_names = f.read().splitlines()
        return cluster_names
    else:
        cluster_names = [cluster_or_filename]
        return cluster_names


def find_path(filename_or_path, data_path):
    """Looks for the likely existing path if given only a filename.

    Args:
        filename_or_path (str): Either the name of a cluster or the name of a file containing cluster names.
        data_path (str): Path to the data directory

    Returns:
        path (str): Path related to the filename.

    """
    in_cwd_path = os.path.join(os.getcwd(), filename_or_path)
    in_custom_data_path = os.path.join(data_path, filename_or_path)
    if os.path.exists(filename_or_path):
        path = filename_or_path
        return path
    elif os.path.exists(in_cwd_path):
        path = in_cwd_path
        return path
    elif os.path.exists(in_custom_data_path):
        path = in_custom_data_path
        return path
    else:
        path = filename_or_path
        return path


def load_cluster_parameters(cluster_parameters_path, cluster_name):
    """Loads in the data of the cluster parameters and returns those of a single cluster.

    Args:
        cluster_parameters_path (str): Path to the (.csv) file where the cluster parameters are saved.
        cluster_name (str): Name of the cluster

    Returns:
        params (dict): Dictionary containing the parameters of a single cluster.

    """
    cluster_params = pd.read_csv(cluster_parameters_path, index_col=0)
    cluster_params = cluster_params[cluster_params['AgeNN'] != '']
    cluster_params = cluster_params.rename(columns={'Cluster': 'name', 'RA_ICRS': 'ra', 'DE_ICRS': 'dec',
                                                    'pmRA_': 'pmra', 'pmDE': 'pmdec', 'e_pmRA_': 'pmra_error',
                                                    'e_pmDE': 'pmdec_error', 'plx': 'parallax',
                                                    'e_plx': 'parallax_error', 'AgeNN': 'age', 'AVNN': 'a_v',
                                                    'DistPc': 'dist'})
    if cluster_name in cluster_params['name'].values:
        params = cluster_params[cluster_params['name'] == cluster_name].to_dict(orient='list')
        params = {key: params[key][0] for key in params}
        return params
    else:
        return None


def load_cone(cone_path, cluster):
    """Loads in the data of the stars found in the cone search. Also performs a number of preprocessing steps.

    Args:
        cone_path (str): Path to the (vot.gz) file where the cone search data is saved.
        cluster (Cluster): Cluster object

    Returns:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.

    """
    cone = parse(cone_path)
    cone = cone.get_first_table().to_table(use_names_over_ids=True)
    cone = cone.to_pandas()

    # Drop sources with incomplete data
    cone = cone.dropna()

    # Correct for the parallax zero-point
    cone['parallax'] += 0.017

    # Correct the magnitude and colour for interstellar extinction
    extinction_correction(cone, cluster)

    # Adjust the galactic coordinates to prevent a split in the sky position plot
    unsplit_sky_positions(cone, coordinate_system='galactic')

    # Calculate magnitude and colour errors from the available flux data
    add_columns([cone], [phot_g_mean_mag_error_function(), bp_rp_error_function()],
                ['phot_g_mean_mag_error', 'bp_rp_error'])

    # Set the default membership probability to zero
    cone['PMemb'] = 0
    return cone


def load_members(members_path, cluster_name, cluster_column='cluster', id_column='source_id', prob_column='PMemb'):
    """Loads in the data of a membership list and selects the members of a single cluster.

    Args:
        members_path (str): Path to the (.csv) file where the member data is saved.
        cluster_name (str): Name of the cluster
        cluster_column (str): Name of the cluster name column in the members dataframe
        id_column (str): Name of the source identity column in the members dataframe
        prob_column (str): Name of the membership probability column in the members dataframe

    Returns:
        cluster_members (Dataframe): Dataframe containing the data of the cluster member sources.

    """
    all_members = pd.read_csv(members_path, index_col=0)
    all_members = all_members.rename(columns={cluster_column: 'cluster', id_column: 'source_id', prob_column: 'PMemb'})

    cluster_members = all_members[all_members['cluster'] == cluster_name]
    cluster_members = cluster_members.reset_index(drop=True)[['source_id', 'PMemb']]
    return cluster_members


def load_isochrone(isochrone_path, cluster):
    """Loads in the data of the isochrones and selects the isochrone of a particular age.
    Also converts the converts the isochrone's absolute G magnitude to apparent magnitude.

    Args:
        isochrone_path (str): Path to the (.dat) file where the isochrone data saved.
        cluster (Cluster): Cluster object

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
    closest_age = ages[np.argmin(np.abs(ages - cluster.age))]
    isochrone = isochrones[isochrones['logAge'] == closest_age]

    # Exclude data beyond the evolutionary stage of the early asymptotic giant branch
    isochrone = isochrone[isochrone['label'] <= 7]

    # Convert to apparent magnitude
    distance_modulus = 5 * np.log10(cluster.dist) - 5
    isochrone['phot_g_mean_mag'] += distance_modulus

    return isochrone


def save_sets(save_dir, members, candidates, non_members, comparison=None):
    """Saves the sets of sources in .csv files.

    Args:
        save_dir (str): Directory where the sets are saved
        members (Dataframe): Dataframe containing member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources
        comparison (Dataframe): Dataframe containing comparison sources

    """
    members.to_csv(os.path.join(save_dir, 'members.csv'))
    candidates.to_csv(os.path.join(save_dir, 'candidates.csv'))
    non_members.to_csv(os.path.join(save_dir, 'non_members.csv'))
    if comparison is not None:
        comparison.to_csv(os.path.join(save_dir, 'comparison.csv'))


def load_sets(save_dir):
    """Loads the sets of sources from .csv files.

    Args:
        save_dir (str): Directory where the sets are loaded from

    Returns:
        members (Dataframe): Dataframe containing member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources
        comparison (Dataframe): Dataframe containing comparison sources

    """
    members = pd.read_csv(os.path.join(save_dir, 'members.csv'))
    candidates = pd.read_csv(os.path.join(save_dir, 'candidates.csv'))
    non_members = pd.read_csv(os.path.join(save_dir, 'non_members.csv'))
    if os.path.exists(os.path.join(save_dir, 'comparison.csv')):
        comparison = pd.read_csv(os.path.join(save_dir, 'comparison.csv'))
    else:
        comparison = None
    return members, candidates, non_members, comparison


def save_cluster(save_dir, cluster):
    """Saves the cluster parameters.

    Args:
        save_dir (str): Directory where the sets are saved
        cluster (Cluster): Cluster object

    """
    with open(os.path.join(save_dir, 'cluster'), 'w') as cluster_file:
        json.dump(vars(cluster), cluster_file)


def load_cluster(save_dir):
    """Loads the cluster parameters.

    Args:
        save_dir (str): Directory where the sets are saved

    Returns:
        cluster (Cluster): Cluster object

    """
    with open(os.path.join(save_dir, 'cluster'), 'r') as cluster_file:
        cluster_params = json.load(cluster_file)
        cluster = Cluster(cluster_params)
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
