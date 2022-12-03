import os
import argparse
import matplotlib
import numpy as np
import pandas as pd

from gaia_oc_amd.io import load_cluster, load_sets, cluster_list, load_model, load_hyper_parameters

from gaia_oc_amd.data_preparation.cluster import Cluster

from gaia_oc_amd.candidate_evaluation.membership_probability import calculate_candidate_probs
from gaia_oc_amd.candidate_evaluation.diagnostics import tidal_radius
from gaia_oc_amd.candidate_evaluation.visualization import plot_density_profile, plot_mass_segregation_profile, \
    plot_venn_diagram, plot_confusion_matrix, plot_sources, plot_sources_limits

PRETRAINED_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'candidate_evaluation/pretrained_model')


def evaluate_clusters(cluster_names, clusters_dir='./data/clusters', model_dir=PRETRAINED_MODEL_DIR, n_samples=100,
                      seed=42, show=False, new_members_label='this study', use_tidal_radius=False,
                      use_comparison_tidal_radius=False, plot_only=False, save_plots=True, fast_mode=False):
    """Main function for evaluating the membership status of cluster candidates. This function contains the following
    steps:
        - Load the model and the hyper parameters
        - Evaluate the membership status of the candidates of a cluster
        - Create a number of plots which give some insight in the new member distribution and compares to either a set
            of training or comparison members.

    Args:
        cluster_names (str, list): 'Names of the open cluster(s) we want to build sets for. Can be a name or a file
            with cluster names.'
        clusters_dir (str): 'Directory where cluster data (e.g. cone searches, source sets) and results will be saved.'
        model_dir (str): 'Directory where the model parameters will be saved.'
        n_samples (int): 'Number of candidate samples to use for calculating the membership probability.'
        seed (int): 'The seed that determines the distribution of train and validation data.'
        show (bool): 'Whether to show the plots.'
        save_plots (bool): 'Whether to save the plots.'
        new_members_label (str): 'Label to use for indicating the training members in plots.'
        use_tidal_radius (bool): 'Whether to calculate the tidal radius of the cluster with the candidates and exclude
            candidates outside the tidal radius from the plots.'
        use_comparison_tidal_radius (bool):'Whether to calculate the tidal radius of the cluster with the comparison
            members and exclude comparison members outside the tidal radius from the plots.'
        plot_only (bool): 'Whether to only create plots for the given clusters. This skips the (re)calculation of
            candidate member probabilities.'
        fast_mode (bool): 'If True, use a faster but more memory intensive method, which might run out of memory for '
                             'many (>~10^6) sources.'
    """

    cluster_names = cluster_list(cluster_names)

    # Load the trained deep sets model
    model = load_model(model_dir)

    # Load dataset hyperparameters
    hyper_parameters = load_hyper_parameters(model_dir)

    source_features = hyper_parameters['source_features']
    source_feature_means = np.array(hyper_parameters['source_feature_means'])
    source_feature_stds = np.array(hyper_parameters['source_feature_stds'])

    cluster_features = hyper_parameters['cluster_features']
    cluster_feature_means = np.array(hyper_parameters['cluster_feature_means'])
    cluster_feature_stds = np.array(hyper_parameters['cluster_feature_stds'])

    size_support_set = hyper_parameters['size_support_set']

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

        if not plot_only:
            calculate_candidate_probs(cluster_dir, model_dir, n_samples=n_samples,
                                      fast_mode=fast_mode, seed=seed)


        print('Creating plots...', end=' ')
        if show or save_plots:
            cluster_params = load_cluster(cluster_dir)
            cluster = Cluster(cluster_params)
            members, candidates, non_members, comparison = load_sets(cluster_dir)

            if cluster.comparison_members_label is not None:
                cluster_comparison_label = cluster.comparison_members_label
            else:
                cluster_comparison_label = 'comparison'

            member_candidates = candidates[candidates['PMemb'] >= 0.1]
            plot_density_profile(member_candidates, cluster, comparison,
                                 save_file=os.path.join(cluster_dir, 'density_profile.png'),
                                 members_label=new_members_label, comparison_label=cluster_comparison_label,
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
                                          members_label=new_members_label, comparison_label=cluster_comparison_label,
                                          title=f'{cluster.name}'.replace('_', ' '),
                                          show=show, save=save_plots)
            plot_confusion_matrix(candidates, comparison,
                                  save_file=os.path.join(cluster_dir, 'membership_comparison.png'),
                                  title=f'{cluster.name}'.replace('_', ' '), label1=new_members_label,
                                  label2=cluster_comparison_label, show=show, save=save_plots)
            plot_venn_diagram(member_candidates, comparison, save_file=os.path.join(cluster_dir, 'venn_diagram.png'),
                              title=f'{cluster.name}'.replace('_', ' '), label1=new_members_label,
                              label2=cluster_comparison_label, show=show, save=save_plots)

            limits = plot_sources_limits(pd.concat((non_members, member_candidates)), cluster.isochrone_colour)

            plot_sources(member_candidates[member_candidates['PMemb'] >= 0.8],
                         save_file=os.path.join(cluster_dir, 'new_members.png'),
                         colour=cluster.isochrone_colour,
                         field_sources=pd.concat((non_members, member_candidates[member_candidates['PMemb'] <
                                                                                 0.8])),
                         members_label=new_members_label + f' ($p\\geq${0.8})',
                         title=f'{cluster.name}'.replace('_', ' '), limits=limits, show=show, save=save_plots)
            plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'comparison.png'),
                         colour=cluster.isochrone_colour,
                         comparison=comparison, plot_type='comparison', members_label=new_members_label,
                         comparison_label=cluster_comparison_label, title=f'{cluster.name}'.replace('_', ' '),
                         limits=limits, show=show, save=save_plots)
            plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'additional_members.png'),
                         colour=cluster.isochrone_colour,
                         comparison=comparison, field_sources=non_members, plot_type='unique_members',
                         members_label=new_members_label, title=f'{cluster.name}'.replace('_', ' '), limits=limits,
                         show_isochrone=True, show_boundaries=True, cluster=cluster, show=show, save=save_plots)
            plot_sources(comparison, save_file=os.path.join(cluster_dir, 'missed_members.png'),
                         colour=cluster.isochrone_colour,
                         comparison=member_candidates, field_sources=non_members, plot_type='unique_members',
                         members_label=cluster_comparison_label, title=f'{cluster.name}'.replace('_', ' '),
                         limits=limits, show_isochrone=True, show_boundaries=True, cluster=cluster, show=show,
                         save=save_plots)
        print(f'done, saved in {os.path.abspath(cluster_dir)}')
        print(' ')
        print(100 * '=')


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
    parser.add_argument('--new_members_label', nargs='?', type=str, default='this study',
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
    parser.add_argument('--fast_mode', nargs='?', type=bool, default=False,
                        help='If True, use a faster but more memory intensive method, which might crash for '
                             'many (>~10^5) sources.')

    args_dict = vars(parser.parse_args())

    evaluate_clusters(args_dict['cluster_names'],
                      clusters_dir=args_dict['clusters_dir'],
                      model_dir=args_dict['model_dir'],
                      n_samples=args_dict['n_samples'],
                      seed=args_dict['seed'],
                      show=args_dict['show'],
                      new_members_label=args_dict['new_members_label'],
                      use_tidal_radius=args_dict['use_tidal_radius'],
                      use_comparison_tidal_radius=args_dict['use_comparison_tidal_radius'],
                      plot_only=args_dict['plot_only'],
                      save_plots=args_dict['save_plots'],
                      fast_mode=args_dict['fast_mode'])
