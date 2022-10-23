import os
import argparse
import matplotlib
import numpy as np
import pandas as pd
from torch import load

from gaia_oc_amd.io import load_cluster, load_sets, cluster_list, load_hyper_parameters

from gaia_oc_amd.machine_learning.deepsets_zaheer import D5

from gaia_oc_amd.candidate_evaluation.membership_probability import calculate_probabilities
from gaia_oc_amd.candidate_evaluation.diagnostics import tidal_radius
from gaia_oc_amd.candidate_evaluation.visualization import plot_density_profile, plot_mass_segregation_profile, \
    plot_venn_diagram, plot_confusion_matrix, plot_sources, plot_sources_limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--clusters_dir', nargs='?', type=str, default='clusters',
                        help='Directory where cluster data is saved.')
    parser.add_argument('--model_dir', nargs='?', type=str, default='deep_sets_model',
                        help='Path to where the model parameters are saved.')
    parser.add_argument('--n_samples', nargs='?', type=int, default=100,
                        help='Number of candidate samples to use for calculating the membership probability.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the sampled features and support set of the candidate.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the plot.')
    parser.add_argument('--save_plots', nargs='?', type=bool, default=True,
                        help='Whether to save the plots.')
    parser.add_argument('--members_label', nargs='?', type=str, default='this study',
                        help='Label to use for indicating the training members in plots.')
    parser.add_argument('--comparison_label', nargs='?', type=str, default=None,
                        help='Label to use for indicating the comparison members in plots.')
    parser.add_argument('--prob_threshold', nargs='?', type=float, default=0.8,
                        help='Minimum membership probability of candidates plotted in the new_members plot.')
    parser.add_argument('--use_tidal_radius', nargs='?', type=bool, default=False,
                        help='Whether to calculate the tidal radius of the cluster with the candidates'
                             'and exclude candidates outside the tidal radius from the plots.')
    parser.add_argument('--use_comparison_tidal_radius', nargs='?', type=bool, default=True,
                        help='Whether to calculate the tidal radius of the cluster '
                             'and exclude candidate members outside the tidal radius from the plots.')
    parser.add_argument('--plot_only', nargs='?', type=bool, default=False,
                        help='Whether to calculate the membership probabilities of the candidates.')

    args_dict = vars(parser.parse_args())

    # main arguments
    clusters_dir = args_dict['clusters_dir']
    cluster_names = cluster_list(args_dict['cluster_names'])

    # path arguments
    model_dir = args_dict['model_dir']

    # evaluation arguments
    n_samples = args_dict['n_samples']
    seed = args_dict['seed']

    # plot arguments
    show = args_dict['show']
    prob_threshold = args_dict['prob_threshold']
    members_label = args_dict['members_label']
    comparison_label = args_dict['comparison_label']
    use_tidal_radius = args_dict['use_tidal_radius']
    use_comparison_tidal_radius = args_dict['use_comparison_tidal_radius']
    plot_only = args_dict['plot_only']

    # save arguments
    save_plots = args_dict['save_plots']

    # Load training and model hyperparameters
    hyper_parameters = load_hyper_parameters(model_dir)
    training_features = hyper_parameters['training_features']
    hidden_size = hyper_parameters['hidden_size']
    training_feature_means = np.array(hyper_parameters['training_feature_means'])
    training_feature_stds = np.array(hyper_parameters['training_feature_stds'])
    size_support_set = hyper_parameters['size_support_set']

    # Load the trained deep sets model
    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)
    model.load_state_dict(load(os.path.join(model_dir, 'model_parameters')))

    n_clusters = len(cluster_names)
    print('Evaluating candidates for:', cluster_names)
    print('Number of clusters:', n_clusters)

    if not show:
        matplotlib.use('Agg')

    for idx, cluster_name in enumerate(cluster_names):
        print(' ')
        print('Cluster:', cluster_name, f' ({idx + 1} / {n_clusters})')

        # Define the cluster data/results directory
        cluster_dir = os.path.join(clusters_dir, cluster_name)

        # Load the necessary datasets (i.e. cluster, sources)
        cluster = load_cluster(cluster_dir)
        members, candidates, non_members, comparison = load_sets(cluster_dir)

        if not plot_only:
            candidate_probabilities = calculate_probabilities(candidates, model, cluster, members,
                                                              training_features, training_feature_means,
                                                              training_feature_stds, size_support_set=size_support_set,
                                                              n_samples=n_samples, seed=seed)

            candidates['PMemb'] = candidate_probabilities
            candidates.to_csv(os.path.join(cluster_dir, 'candidates.csv'))

        print('Creating plots...', end=' ')
        if show or save_plots:

            if comparison_label is None and cluster.comparison_members_label is not None:
                comparison_label = cluster.comparison_members_label

            member_candidates = candidates[candidates['PMemb'] >= 0.1]
            plot_density_profile(member_candidates, cluster, comparison,
                                 save_file=os.path.join(cluster_dir, 'density_profile.png'),
                                 members_label=members_label, comparison_label=comparison_label,
                                 title=f'{cluster.name}'.replace('_', ' '),
                                 show=show, save=save_plots)

            # If we do not train on the sky position feature (f_r), we can use the tidal radius to constrain the members
            if use_tidal_radius:
                r_t = tidal_radius(member_candidates, cluster)
                non_members = pd.concat((non_members, candidates[(candidates['PMemb'] < 0.1) |
                                                                 (candidates['f_r'] > r_t)]))
                member_candidates = member_candidates[member_candidates['f_r'] <= r_t]
            else:
                non_members = pd.concat((non_members, candidates[candidates['PMemb'] < 0.1]))

            if use_comparison_tidal_radius:
                comparison = comparison[comparison['f_r'] <= tidal_radius(comparison, cluster)]

            plot_mass_segregation_profile(member_candidates, cluster, comparison,
                                          save_file=os.path.join(cluster_dir, 'mass_segregation.png'),
                                          members_label=members_label, comparison_label=comparison_label,
                                          title=f'{cluster.name}'.replace('_', ' '),
                                          show=show, save=save_plots)
            plot_confusion_matrix(candidates, comparison,
                                  save_file=os.path.join(cluster_dir, 'membership_comparison.png'),
                                  title=f'{cluster.name}'.replace('_', ' '), label1=members_label,
                                  label2=comparison_label, show=show, save=save_plots)
            plot_venn_diagram(member_candidates, comparison, save_file=os.path.join(cluster_dir, 'venn_diagram.png'),
                              title=f'{cluster.name}'.replace('_', ' '), label1=members_label, label2=comparison_label,
                              show=show, save=save_plots)

            limits = plot_sources_limits(pd.concat((non_members, member_candidates)))

            plot_sources(member_candidates[member_candidates['PMemb'] >= 0.8],
                         save_file=os.path.join(cluster_dir, 'new_members.png'),
                         field_sources=pd.concat((non_members, member_candidates[member_candidates['PMemb'] <
                                                                                 0.8])),
                         members_label=members_label + f' ($p\\geq${0.8})',
                         title=f'{cluster.name}'.replace('_', ' '), limits=limits, show=show, save=save_plots)
            plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'comparison.png'),
                         comparison=comparison, plot_type='comparison', members_label=members_label,
                         comparison_label=comparison_label, title=f'{cluster.name}'.replace('_', ' '), limits=limits,
                         show=show, save=save_plots)
            plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'additional_members.png'),
                         comparison=comparison, field_sources=non_members, plot_type='unique_members',
                         members_label=members_label, title=f'{cluster.name}'.replace('_', ' '), limits=limits,
                         show_isochrone=True, show_boundaries=True, cluster=cluster, show=show, save=save_plots)
            plot_sources(comparison, save_file=os.path.join(cluster_dir, 'missed_members.png'),
                         comparison=member_candidates, field_sources=non_members, plot_type='unique_members',
                         members_label=comparison_label, title=f'{cluster.name}'.replace('_', ' '), limits=limits,
                         show_isochrone=True, show_boundaries=True, cluster=cluster, show=show, save=save_plots)
        print(f'done, saved in {os.path.abspath(cluster_dir)}')
        print(' ')
        print(100 * '=')
