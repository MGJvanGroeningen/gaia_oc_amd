import os
import argparse
from torch import load

from gaia_oc_amd.data_preparation.isochrone import make_isochrone
from gaia_oc_amd.data_preparation.features import Features
from gaia_oc_amd.candidate_evaluation.probabilities import candidate_probabilities
from gaia_oc_amd.candidate_evaluation.diagnostics import save_tidal_radius
from gaia_oc_amd.data_preparation.sets import Sources
from gaia_oc_amd.neural_networks.deepsets_zaheer import D5
from gaia_oc_amd.candidate_evaluation.visualization import make_plots
from gaia_oc_amd.data_preparation.utils import load_sets, cluster_list, find_path

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
    parser.add_argument('--model_parameters_save_file', nargs='?', type=str, default='deep_sets_model_parameters',
                        help='Path to where the model parameters will be saved.')
    parser.add_argument('--size_support_set', nargs='?', type=int, default=5,
                        help='The number of members in the support set of the candidate samples.')
    parser.add_argument('--training_features', nargs='*', type=str, default=['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g'],
                        help='Features on which the model was be trained.')
    parser.add_argument('--members_label', nargs='?', type=str, default='train',
                        help='Label to use for indicating the training members in plots.')
    parser.add_argument('--comparison_label', nargs='?', type=str, default='comparison',
                        help='Label to use for indicating the comparison members in plots.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the features and support set of the candidate samples.')
    parser.add_argument('--hidden_size', nargs='?', type=int, default=64,
                        help='Hidden size of the neural network layers.')
    parser.add_argument('--n_samples', nargs='?', type=int, default=40,
                        help='Number of candidate samples to use for calculating the membership probability.')
    parser.add_argument('--prob_threshold', nargs='?', type=float, default=0.8,
                        help='Minimum membership probability of candidates plotted in the new_members plot.')
    parser.add_argument('--use_tidal_radius', nargs='?', type=bool, default=False,
                        help='Whether to calculate the tidal radius of the cluster '
                             'and exclude candidate members outside the tidal radius from the plot.')
    parser.add_argument('--show_train_members', nargs='?', type=bool, default=False,
                        help='Whether to show the training members on top of/instead of the comparison members.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the plot.')
    parser.add_argument('--save_plots', nargs='?', type=bool, default=True,
                        help='Whether to save the plots.')

    args_dict = vars(parser.parse_args())

    data_dir = args_dict['data_dir']
    cluster_names = cluster_list(args_dict['cluster_names'], data_dir)
    isochrone_path = find_path(args_dict['isochrone_path'], data_dir)
    model_parameters_save_file = find_path(args_dict['model_parameters_save_file'], data_dir)

    size_support_set = args_dict['size_support_set']
    training_features = args_dict['training_features']
    members_label = args_dict['members_label']
    comparison_label = args_dict['comparison_label']
    seed = args_dict['seed']

    hidden_size = args_dict['hidden_size']

    n_samples = args_dict['n_samples']

    show = args_dict['show']
    save_plots = args_dict['save_plots']
    prob_threshold = args_dict['prob_threshold']
    use_tidal_radius = args_dict['use_tidal_radius']
    show_train_members = args_dict['show_train_members']

    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)
    model.load_state_dict(load(model_parameters_save_file))

    print('Evaluating candidates for:', cluster_names)
    print('Number of clusters:', len(cluster_names))

    for cluster_name in cluster_names:
        print(' ')
        print('Cluster:', cluster_name)
        cluster, members, candidates, non_members, comparison = load_sets(data_dir, cluster_name)
        isochrone = make_isochrone(isochrone_path, cluster)

        save_dir = os.path.join(data_dir, 'clusters', cluster.name)

        features = Features(training_features, cluster, isochrone)
        sources = Sources(members, candidates, non_members, comparison_members=comparison,
                          members_label=members_label, comparison_label=comparison_label)

        sources.candidates['PMemb'] = candidate_probabilities(model, sources, features, n_samples,
                                                              size_support_set=size_support_set, seed=seed)
        sources.candidates.to_csv(os.path.join(save_dir, 'candidates.csv'))

        if use_tidal_radius:
            save_tidal_radius(data_dir, sources.candidates.hp(min_prob=0.1), cluster)
            tidal_radius = cluster.r_t
        else:
            tidal_radius = None

        print('Creating plots...', end=' ')
        if show or save_plots:
            make_plots(sources, cluster, save_dir, isochrone, prob_threshold=prob_threshold, show=show, save=save_plots,
                       tidal_radius=tidal_radius, show_train_members=show_train_members)
        print(f'done, saved in {os.path.abspath(save_dir)}')
