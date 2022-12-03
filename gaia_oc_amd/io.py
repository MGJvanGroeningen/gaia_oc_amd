import os
import json
import numpy as np
import pandas as pd
from astropy.io.votable import parse
from torch import load, save

from gaia_oc_amd.machine_learning.deepsets_zaheer import D5


def cluster_list(cluster_list_source):
    """Function for creating a list of cluster names. If the input of this function is a file path,
    return a list of the lines in the file. If it is a string, return a list with only the input string.

    Args:
        cluster_list_source (str, list): Either the name of a cluster, the name of a file containing cluster names or a
            list of cluster names.

    Returns:
        cluster_names (str, list): List of cluster names.

    """
    if type(cluster_list_source) == str:
        if os.path.exists(cluster_list_source):
            # cluster_list_source should be a path to a file with cluster names
            with open(cluster_list_source, 'r') as f:
                cluster_names = f.read().splitlines()
            return cluster_names
        else:
            # cluster_list_source should be a single cluster name
            cluster_names = [cluster_list_source]
            return cluster_names
    elif type(cluster_list_source) == list:
        # cluster_list_source should be a list of cluster names
        cluster_names = cluster_list_source
        return cluster_names
    else:
        raise TypeError(f'Wrong input type: {type(cluster_list_source)}, use a string or list.')


def load_cluster_parameters(cluster_parameters_path, cluster_name, new_column_names=None):
    """Loads in the data of the cluster parameters and returns those of a single cluster.

    Args:
        cluster_parameters_path (str): Path to the (.csv) file where the cluster parameters are saved.
        cluster_name (str): Name of the cluster
        new_column_names (dict): Dictionary for renaming the dataframe columns in the format
            {'old_name': 'new_name', etc...}

    Returns:
        params (dict): Dictionary containing the parameters of a single cluster.

    """
    cluster_params = pd.read_csv(cluster_parameters_path, index_col=0).dropna()

    if new_column_names is not None:
        cluster_params = cluster_params.rename(columns=new_column_names)
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


def load_isochrone(isochrone_path, age):
    """Loads in the data of the isochrones and selects the isochrone of a particular age.

    Args:
        isochrone_path (str): Path to the (.dat) file where the isochrone data saved.
        age (float): log(age) of the isochrone

    Returns:
        isochrone (Dataframe): Dataframe containing isochrone data points.

    """
    isochrones = pd.read_csv(isochrone_path,
                             names=['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg',
                                    'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3',
                                    'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess',
                                    'Z', 'mbolmag', 'phot_g_mean_mag', 'G_BPmag', 'G_RPmag'],
                             delim_whitespace=True,
                             comment='#')

    # Select the isochrone with the age of the cluster
    ages = np.array(list(set(isochrones['logAge'].values)))
    closest_age = ages[np.argmin(np.abs(ages - age))]
    age_error = np.abs(closest_age - age)
    if age_error > 0.02:
        raise UserWarning(f'The isochrone in the data file with age {closest_age} is closest to the supplied age '
                          f'{age}. A large difference might result in an isochrone that poorly fits the member '
                          f'distribution.')
    isochrone = isochrones[isochrones['logAge'] == closest_age]
    return isochrone


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
    # with open(os.path.join(save_dir, 'cluster'), 'wb') as cluster_file:
    #     pickle.dump(cluster, cluster_file)
    with open(os.path.join(save_dir, 'cluster'), 'w') as cluster_params_file:
        cluster_params = vars(cluster).copy()
        if cluster_params['isochrone'] is not None:
            cluster_params['isochrone'] = cluster_params['isochrone'].to_numpy().T.tolist()
        json.dump(cluster_params, cluster_params_file)


def load_cluster(save_dir):
    """Loads the cluster parameters.

    Args:
        save_dir (str): Directory where the sets are saved

    Returns:
        cluster (Cluster): Cluster object

    """
    # with open(os.path.join(save_dir, 'cluster'), 'rb') as cluster_file:
    #     cluster = pickle.load(cluster_file)
    with open(os.path.join(save_dir, 'cluster'), 'r') as cluster_file:
        cluster_params = json.load(cluster_file)
    if cluster_params['isochrone'] is not None:
        cluster_params['isochrone'] = pd.DataFrame({cluster_params['isochrone_colour']: cluster_params['isochrone'][0],
                                                    'phot_g_mean_mag': cluster_params['isochrone'][1]})
    return cluster_params


def save_hyper_parameters(model_save_dir, hyper_parameters):
    """Saves the hyperparameters of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model will be saved
        hyper_parameters (dict): Dictionary containing the data and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'hyper_parameters'), 'w') as f:
        json.dump(hyper_parameters, f)


def load_hyper_parameters(model_save_dir):
    """Loads the hyperparameters of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model are stored

    Returns:
        hyper_parameters (dict): Dictionary containing the data and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'hyper_parameters'), 'r') as f:
        hyper_parameters = json.load(f)
    return hyper_parameters


def save_metrics(model_save_dir, metrics_dict):
    """Saves the metrics of a certain model.

    Args:
        model_save_dir (str): Directory where the hyperparameters corresponding to a specific model will be saved
        metrics_dict (dict): Dictionary containing and training hyperparameters

    """
    with open(os.path.join(model_save_dir, 'metrics'), 'w') as f:
        json.dump(metrics_dict, f)


def load_metrics(model_save_dir):
    """Loads the metrics of a certain model.

    Args:
        model_save_dir (str): Directory where the metrics corresponding to a specific model are stored

    Returns:
        metrics (dict): Dictionary containing and training metrics

    """
    with open(os.path.join(model_save_dir, 'metrics'), 'r') as f:
        metrics_dict = json.load(f)
    return metrics_dict


def save_model(model_save_dir, model):
    """Saves the (trained) parameters of a deep sets model.

    Args:
        model_save_dir (str): Directory where the parameters corresponding to a specific model will be saved
        model (D5): Deep sets model with (trained) parameters

    """
    save(model.state_dict(), os.path.join(model_save_dir, 'model_parameters'))


def load_model(model_save_dir):
    """Loads a pretrained model with the (hyper)parameters in model_save_dir.

    Args:
        model_save_dir (str): Directory where the model data is stored

    Returns:
        model (D5): Model with pretrained parameters

    """
    hyper_parameters = load_hyper_parameters(model_save_dir)
    source_features = hyper_parameters['source_features']
    cluster_features = hyper_parameters['cluster_features']
    hidden_size = hyper_parameters['hidden_size']

    model = D5(hidden_size, x_dim=2 * len(source_features) + len(cluster_features), pool='mean', out_dim=2)
    model.load_state_dict(load(os.path.join(model_save_dir, 'model_parameters')))
    return model
