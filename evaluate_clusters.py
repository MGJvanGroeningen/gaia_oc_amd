import os
import argparse
import numpy as np
from torch import load

from gaia_oc_amd.data_preparation.sets import Sources
from gaia_oc_amd.data_preparation.io import load_cluster, load_sets, cluster_list, find_path, load_hyper_parameters, \
    load_isochrone, save_cluster

from gaia_oc_amd.neural_networks.deepsets_zaheer import D5

from gaia_oc_amd.candidate_evaluation.probabilities import calculate_probabilities
from gaia_oc_amd.candidate_evaluation.diagnostics import tidal_radius
from gaia_oc_amd.candidate_evaluation.visualization import make_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')
    parser.add_argument('--isochrone_path', nargs='?', type=str, default='isochrones.dat',
                        help='Path for retrieving isochrone data. Expects a .dat file with isochrone data '
                             'between log(age) of 6 and 10.')
    parser.add_argument('--model_save_dir', nargs='?', type=str, default='deep_sets_model',
                        help='Path to where the model parameters will be saved.')
    parser.add_argument('--n_samples', nargs='?', type=int, default=40,
                        help='Number of candidate samples to use for calculating the membership probability.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the features and support set of the candidate samples.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the plot.')
    parser.add_argument('--save_plots', nargs='?', type=bool, default=True,
                        help='Whether to save the plots.')
    parser.add_argument('--members_label', nargs='?', type=str, default='train',
                        help='Label to use for indicating the training members in plots.')
    parser.add_argument('--comparison_label', nargs='?', type=str, default='comparison',
                        help='Label to use for indicating the comparison members in plots.')
    parser.add_argument('--prob_threshold', nargs='?', type=float, default=0.8,
                        help='Minimum membership probability of candidates plotted in the new_members plot.')
    parser.add_argument('--use_tidal_radius', nargs='?', type=bool, default=False,
                        help='Whether to calculate the tidal radius of the cluster '
                             'and exclude candidate members outside the tidal radius from the plot.')
    parser.add_argument('--compare_train_members', nargs='?', type=bool, default=False,
                        help='Whether to compare against the training members on top of/instead of '
                             'the comparison members.')

    args_dict = vars(parser.parse_args())

    # main arguments
    data_dir = args_dict['data_dir']
    cluster_names = cluster_list(args_dict['cluster_names'], data_dir)

    # path arguments
    isochrone_path = find_path(args_dict['isochrone_path'], data_dir)
    model_save_dir = find_path(args_dict['model_save_dir'], data_dir)

    # evaluation arguments
    n_samples = args_dict['n_samples']
    seed = args_dict['seed']

    # plot arguments
    show = args_dict['show']
    prob_threshold = args_dict['prob_threshold']
    members_label = args_dict['members_label']
    comparison_label = args_dict['comparison_label']
    use_tidal_radius = args_dict['use_tidal_radius']
    compare_train_members = args_dict['compare_train_members']

    # save arguments
    save_plots = args_dict['save_plots']

    hyper_parameters = load_hyper_parameters(model_save_dir)
    training_features = hyper_parameters['training_features']
    hidden_size = hyper_parameters['hidden_size']
    training_feature_means = np.array(hyper_parameters['training_feature_means'])
    training_feature_stds = np.array(hyper_parameters['training_feature_stds'])
    size_support_set = hyper_parameters['size_support_set']

    # Load the trained deep sets model
    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)
    model.load_state_dict(load(os.path.join(model_save_dir, 'model_parameters')))

    n_clusters = len(cluster_names)
    print('Evaluating candidates for:', cluster_names)
    print('Number of clusters:', n_clusters)

    for idx, cluster_name in enumerate(cluster_names):
        print(' ')
        print('Cluster:', cluster_name, f' ({idx + 1} / {n_clusters})')

        # Define the cluster data/results directory
        cluster_dir = os.path.join(data_dir, 'clusters', cluster_name)

        # Load the necessary datasets (i.e. cluster, sources and isochrone)
        members, candidates, non_members, comparison = load_sets(cluster_dir)
        cluster = load_cluster(cluster_dir)
        isochrone = load_isochrone(isochrone_path, cluster)

        candidate_probabilities = calculate_probabilities(candidates, model, cluster, isochrone, members,
                                                          training_features, training_feature_means,
                                                          training_feature_stds, size_support_set=size_support_set,
                                                          n_samples=n_samples, seed=seed)

        candidates['PMemb'] = candidate_probabilities
        candidates.to_csv(os.path.join(cluster_dir, 'candidates.csv'))

        # If we do not train on the sky position feature (f_r), we can use the tidal radius to constrain the new members
        if use_tidal_radius:
            r_t = tidal_radius(candidates[candidates['PMemb'] >= 0.1], cluster)
            cluster.r_t = r_t
            save_cluster(data_dir, cluster)
        else:
            r_t = None

        sources = Sources(members, candidates, non_members, comparison=comparison, members_label=members_label,
                          comparison_label=comparison_label)

        print('Creating plots...', end=' ')
        if show or save_plots:
            make_plots(sources, cluster, cluster_dir, isochrone, prob_threshold=prob_threshold, tidal_radius=r_t,
                       compare_train_members=compare_train_members, show=show, save=save_plots)
        print(f'done, saved in {os.path.abspath(cluster_dir)}')
        print(' ')
        print(150 * '=')
