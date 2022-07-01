import os
import json
import numpy as np
import pandas as pd
from astropy.io.votable import parse

from gaia_oc_amd.data_preparation.cluster import Cluster


def cluster_list(cluster_or_filename, data_path):
    path = find_path(cluster_or_filename, data_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            cluster_names = f.read().splitlines()
        return cluster_names
    else:
        return [cluster_or_filename]


def find_path(path, data_path):
    in_cwd_path = os.path.join(os.getcwd(), path)
    in_custom_data_path = os.path.join(data_path, path)
    if os.path.exists(path):
        return path
    elif os.path.exists(in_cwd_path):
        return in_cwd_path
    elif os.path.exists(in_custom_data_path):
        return in_custom_data_path
    else:
        return path


def normalize(dataframe, labels_not_normalized=None, all_data=None):
    if all_data is not None:
        data_norm = (dataframe - all_data.mean()) / all_data.std()
    else:
        data_norm = (dataframe - dataframe.mean()) / dataframe.std()
    if labels_not_normalized is not None:
        data_norm[labels_not_normalized] = dataframe[labels_not_normalized]
    return data_norm


def norm(arr, axis=0):
    return np.sqrt(np.sum(np.square(arr), axis=axis))


def danielski_g(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def danielski_bp(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def danielski_rp(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def extinction_correction(sources, cluster):
    if 'a0' in sources.columns.to_list():
        a_v = sources['a0']
    else:
        a_v = cluster.a_v

    extinction_g = danielski_g(a_v, sources['bp_rp'])
    extinction_bp = danielski_bp(a_v, sources['bp_rp'])
    extinction_rp = danielski_rp(a_v, sources['bp_rp'])

    sources['phot_g_mean_mag'] -= extinction_g
    sources['bp_rp'] -= extinction_bp - extinction_rp
    return sources


def undo_split(df, coordinates='icrs'):
    if coordinates == 'icrs':
        feat = 'ra'
        x = df['ra']
    elif coordinates == 'galactic':
        feat = 'l'
        x = df['l']
    else:
        feat = None
        x = None

    if x is not None:
        if x.max() - x.min() > 180:
            ra = np.where(x > 180, x - 360, x)
            df[feat] = ra


def load_cone(cone_path, cluster):
    cone = parse(cone_path)
    cone = cone.get_first_table().to_table(use_names_over_ids=True)
    cone = cone.to_pandas().dropna()

    cone['parallax'] += 0.017
    extinction_correction(cone, cluster)
    undo_split(cone, coordinates='galactic')
    cone['PMemb'] = 0
    return cone


def load_members(members_path, cluster_name, cluster_column='cluster', id_column='source_id', prob_column='PMemb'):
    all_members = pd.read_csv(members_path, index_col=0)
    all_members = all_members.rename(columns={cluster_column: 'cluster', id_column: 'source_id', prob_column: 'PMemb'})

    cluster_members = all_members[all_members['cluster'] == cluster_name]
    cluster_members = cluster_members.reset_index(drop=True)[['source_id', 'PMemb']]
    return cluster_members


def load_cluster_parameters(cluster_name, cluster_parameters_path):
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


def save_sets(data_dir, cluster, members, candidates, non_members, comparison=None):
    save_dir = os.path.join(data_dir, 'clusters', cluster.name)
    with open(os.path.join(save_dir, 'cluster'), 'w') as cluster_file:
        json.dump(vars(cluster), cluster_file)
    members.to_csv(os.path.join(save_dir, 'members.csv'))
    candidates.to_csv(os.path.join(save_dir, 'candidates.csv'))
    non_members.to_csv(os.path.join(save_dir, 'non_members.csv'))
    if comparison is not None:
        comparison.to_csv(os.path.join(save_dir, 'comparison.csv'))


def load_sets(data_dir, cluster_name):
    save_dir = os.path.join(data_dir, 'clusters', cluster_name)
    members = pd.read_csv(os.path.join(save_dir, 'members.csv'))
    candidates = pd.read_csv(os.path.join(save_dir, 'candidates.csv'))
    non_members = pd.read_csv(os.path.join(save_dir, 'non_members.csv'))
    if os.path.exists(os.path.join(save_dir, 'comparison.csv')):
        comparison = pd.read_csv(os.path.join(save_dir, 'comparison.csv'))
    else:
        comparison = None
    with open(os.path.join(save_dir, 'cluster'), 'r') as cluster_file:
        cluster_params = json.load(cluster_file)
        cluster = Cluster(cluster_params)
    return cluster, members, candidates, non_members, comparison
