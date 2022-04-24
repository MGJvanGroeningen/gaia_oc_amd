import os
import pandas as pd
from sets import Sources


def members_path_to_ref(path):
    if path.split('/')[-2] == 'cg18_members':
        ref = 'Cantat-Gaudin+18'
    elif path.split('/')[-2] == 't22_members':
        ref = 'Tarricq+22'
    else:
        raise ValueError(f'No reference could be deducted from the following path: {path}')
    return ref


def get_data_and_save_dir(model_type):
    if os.getcwd().split('/')[2] == 'mvgroeningen':
        data_dir = '/data1/mvgroeningen/amd/data'
        save_dir = '/data1/mvgroeningen/amd/' + model_type
        # data_dir = '/data1/mvgroeningen/amd/data'
        # save_dir = '/data1/mvgroeningen/amd/' + model_type
        import matplotlib
        matplotlib.use('Agg')
        print('Running on strw')
    else:
        data_dir = os.path.join(os.getcwd(), 'data')
        save_dir = os.path.join(os.getcwd(), model_type)
        print('Running at home')
    return data_dir, save_dir


def save_dirs(save_dir, cluster_name):
    results_dir = os.path.join(save_dir, 'results')
    saved_models_dir = os.path.join(save_dir, 'saved_models')
    log_dir = os.path.join(save_dir, 'log_dir')
    cluster_save_dir = os.path.join(results_dir, cluster_name)

    for directory in [results_dir, saved_models_dir, cluster_save_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    save_paths = {'model': saved_models_dir,
                  'cluster': cluster_save_dir,
                  'log': log_dir}

    return save_paths


def data_dirs(data_dir, cluster_name, train_set, comparison_set):
    credentials_path = os.path.join(data_dir, 'gaia_credentials')
    cone_path = os.path.join(data_dir, 'cones', cluster_name + '.vot.gz')
    train_members_path = os.path.join(data_dir, f'{train_set}_members', cluster_name + '.csv')
    comparison_members_path = os.path.join(data_dir, f'{comparison_set}_members', cluster_name + '.csv')

    isochrone_path = os.path.join(data_dir, 'isochrones')
    cluster_path = os.path.join(data_dir, 'cluster_parameters.tsv')

    data_paths = {'cone': cone_path,
                  'train_members': train_members_path,
                  'comparison_members': comparison_members_path,
                  'isochrone': isochrone_path,
                  'cluster': cluster_path,
                  'credentials': credentials_path}

    return data_paths


def save_csv(sources, cluster_save_dir, combined=True):

    sources_list = [sources.train_members, sources.candidates, sources.noise, sources.comparison_members]
    ids = [0, 1, 2, 3]
    source_labels = ['members', 'candidates', 'noise', 'comparison']

    sets_save_file = os.path.join(cluster_save_dir, 'sets.csv')

    if combined:
        for source, idx in zip(sources_list, ids):
            source['id'] = idx

        sets_df = pd.concat(sources_list, sort=False, ignore_index=True)
        sets_df['train_members_ref'] = sources.train_members_ref
        sets_df['comparison_members_ref'] = sources.comparison_members_ref
        sets_df['prob_threshold'] = sources.prob_threshold
        sets_df.to_csv(sets_save_file)
    else:
        for source, label in zip(sources_list, source_labels):
            source.to_csv(os.path.join(cluster_save_dir, f'{label}.csv'))


def load_csv(cluster_save_dir):
    sets_file = os.path.join(cluster_save_dir, 'sets.csv')

    sets = pd.read_csv(sets_file)

    train_members_ref = sets['train_members_ref'][0]
    comparison_members_ref = sets['comparison_members_ref'][0]
    prob_threshold = sets['prob_threshold'][0]

    members = sets[sets['id'] == 0].copy()
    candidates = sets[sets['id'] == 1].copy()
    noise = sets[sets['id'] == 2].copy()
    comparison = sets[sets['id'] == 3].copy()

    sources = Sources(members, comparison, candidates, noise, prob_threshold, train_members_ref, comparison_members_ref)
    return sources
