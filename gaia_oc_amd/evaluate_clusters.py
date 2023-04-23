import os
import argparse
import matplotlib
import pandas as pd

from gaia_oc_amd.io import load_cluster, load_sets, cluster_list

from gaia_oc_amd.data_preparation.cluster import Cluster

from gaia_oc_amd.candidate_evaluation.membership_probability import calculate_candidate_probs
from gaia_oc_amd.candidate_evaluation.diagnostics import tidal_radius
from gaia_oc_amd.candidate_evaluation.visualization import plot_density_profile, plot_mass_segregation_profile, \
    plot_venn_diagram, plot_confusion_matrix, plot_sources, plot_sources_limits

PRETRAINED_MODEL_DIRS = {'v01': os.path.join(os.path.dirname(__file__),
                                             'candidate_evaluation/pretrained_models/DS10_v01'),
                         'v02': os.path.join(os.path.dirname(__file__),
                                             'candidate_evaluation/pretrained_models/DS10_v02')}


def evaluate_clusters(cluster_names, clusters_dir='./data/clusters', model_dir=PRETRAINED_MODEL_DIRS['v02'],
                      n_samples=100, size_support_set=10, hard_size_ss=True, new_members_label='this study',
                      candidate_prob_threshold=0.1, new_members_prob_threshold=0.8, use_tidal_radius=False,
                      use_comparison_tidal_radius=False, new_members_plot=True, comparison_plot=True,
                      additional_members_plot=False, missed_members_plot=False, venn_diagram_plot=False,
                      confusion_matrix_plot=False, density_profile_plot=False, mass_segregation_plot=False,
                      plot_stellar_field=True, plot_only=False, show_plots=False, save_plots=True, seed=42):
    """Main function for evaluating the membership status of cluster candidates. This function contains the following
    steps:
        - Load the model and the hyper parameters
        - Evaluate the membership status of the candidates of a cluster
        - Create a number of plots which give some insight in the new member distribution and compares to either a set
            of training or comparison members.

    Args:
        cluster_names (str, list): 'Names of the open cluster(s) we want to build sets for. Can be a name or a file
            with cluster names.'
        clusters_dir (str): 'Directory where cluster data and results are saved.'
        model_dir (str): 'Directory of the model, which contains its (hyper)parameters, that will be used
            to evaluate the candidate sources.'
        n_samples (int): 'Number of candidate samples to use for calculating the membership probability.'
        size_support_set (int): 'The number of members in the support set.'
        hard_size_ss (bool): 'When false, set the support set size to the number of available training members when the
            former is larger than the latter.'
        new_members_label (str): 'Label to use for indicating the training members in plots.'
        candidate_prob_threshold (float): 'Minimum membership probability of candidates plotted in the comparison and
            additional members plot.'
        new_members_prob_threshold (float): 'Minimum membership probability of candidates plotted in the new members
            plot.'
        use_tidal_radius (bool): 'Whether to calculate the tidal radius of the cluster with the candidates and exclude
            candidates outside the tidal radius from the plots.'
        use_comparison_tidal_radius (bool): 'Whether to calculate the tidal radius of the cluster with the comparison
            members and exclude comparison members outside the tidal radius from the plots.'
        new_members_plot (bool): 'Whether to create a plot showing the candidates above a threshold probability.'
        comparison_plot (bool): 'Whether to create a plot showing the candidates and comparison members above a
            threshold probability.'
        additional_members_plot (bool): 'Whether to create a plot showing the candidates above a threshold probability,
            which are not present in the comparison members.'
        missed_members_plot (bool): 'Whether to create a plot showing the comparison members, which are not present
            among the candidates above a threshold probability.'
        venn_diagram_plot (bool): 'Whether to create a plot showing a Venn diagram comparing the candidates and the
            comparison members above 10%, 50% and 90% membership probability.'
        confusion_matrix_plot (bool): 'Whether to create a plot showing how the candidate probability compares against
            the probability given in the comparison. A 'confusion matrix' is created by using probability bins.'
        density_profile_plot (bool): 'Whether to create a plot showing the density profile of the candidates above a
            threshold probability and the comparison members.'
        mass_segregation_plot (bool): 'Whether to create a plot showing the mass segregation of the candidates above a
            threshold probability and the comparison members.'
        plot_stellar_field (bool): 'Whether to add the stellar field to the sources plots.'
        plot_only (bool): 'Whether to only create plots for the given clusters. This skips the (re)calculation of
            candidate member probabilities.'
        show_plots (bool): 'Whether to show the plots.'
        save_plots (bool): 'Whether to save the plots.'
        seed (int): 'The seed that determines the sampling of sources when determining membership probabilities.'
    """

    cluster_names = cluster_list(cluster_names)

    n_clusters = len(cluster_names)
    print('Evaluating candidates for:', cluster_names)
    print('Number of clusters:', n_clusters)

    if not show_plots:
        matplotlib.use('Agg')

    for idx, cluster_name in enumerate(cluster_names):
        print(' ')
        print('Cluster:', cluster_name, f' ({idx + 1} / {n_clusters})')

        # Define the cluster data/results directory
        cluster_dir = os.path.join(clusters_dir, cluster_name)

        if not plot_only:
            calculate_candidate_probs(cluster_dir, model_dir, n_samples=n_samples, size_support_set=size_support_set,
                                      hard_size_ss=hard_size_ss, seed=seed)

        print('Creating plots...', end=' ')
        if show_plots or save_plots:
            cluster_params = load_cluster(cluster_dir)
            cluster = Cluster(cluster_params)
            members, candidates, non_members, comparison = load_sets(cluster_dir)

            if cluster.comparison_members_label is not None:
                cluster_comparison_label = cluster.comparison_members_label
            else:
                cluster_comparison_label = 'comparison'

            member_candidates = candidates.query(f'PMemb >= {candidate_prob_threshold}')
            non_member_candidates = candidates.query(f'PMemb < {candidate_prob_threshold}')
            field_sources = None

            if density_profile_plot:
                plot_density_profile(member_candidates, cluster, comparison,
                                     save_file=os.path.join(cluster_dir, 'density_profile.png'),
                                     members_label=new_members_label, comparison_label=cluster_comparison_label,
                                     title=f'{cluster.name}'.replace('_', ' '),
                                     show=show_plots, save=save_plots)

            # If we do not train on the sky position feature (f_r), we can use the tidal radius to constrain the members
            if use_tidal_radius:
                r_t = tidal_radius(member_candidates, cluster)
                member_candidates = member_candidates.query(f'f_r <= {r_t}')
                non_member_candidates = candidates.query(f'(PMemb < {candidate_prob_threshold}) or (f_r > {r_t})')

            if use_comparison_tidal_radius:
                comparison = comparison.query(f'f_r <= {tidal_radius(comparison, cluster)}')

            if plot_stellar_field:
                field_sources = pd.concat((non_members, non_member_candidates))

            if mass_segregation_plot:
                plot_mass_segregation_profile(member_candidates, cluster, comparison,
                                              save_file=os.path.join(cluster_dir, 'mass_segregation.png'),
                                              members_label=new_members_label,
                                              comparison_label=cluster_comparison_label,
                                              title=f'{cluster.name}'.replace('_', ' '), show=show_plots,
                                              save=save_plots)
            if confusion_matrix_plot:
                plot_confusion_matrix(candidates, comparison,
                                      save_file=os.path.join(cluster_dir, 'membership_comparison.png'),
                                      title=f'{cluster.name}'.replace('_', ' '), label1=new_members_label,
                                      label2=cluster_comparison_label, show=show_plots, save=save_plots)
            if venn_diagram_plot:
                plot_venn_diagram(member_candidates, comparison,
                                  save_file=os.path.join(cluster_dir, 'venn_diagram.png'),
                                  title=f'{cluster.name}'.replace('_', ' '), label1=new_members_label,
                                  label2=cluster_comparison_label, show=show_plots, save=save_plots)

            limits = plot_sources_limits(pd.concat((field_sources, member_candidates)), cluster.isochrone_colour)

            if new_members_plot:
                new_members = member_candidates.query(f'PMemb >= {new_members_prob_threshold}')
                plot_sources(new_members, save_file=os.path.join(cluster_dir, 'new_members.png'),
                             colour=cluster.isochrone_colour, field_sources=field_sources,
                             members_label=new_members_label + f' ($p\\geq${new_members_prob_threshold})',
                             title=f'{cluster.name}'.replace('_', ' '), limits=limits, show=show_plots, save=save_plots)
            if comparison_plot:
                plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'comparison.png'),
                             colour=cluster.isochrone_colour, field_sources=field_sources, comparison=comparison,
                             plot_type='comparison', members_label=new_members_label,
                             comparison_label=cluster_comparison_label, title=f'{cluster.name}'.replace('_', ' '),
                             limits=limits, show=show_plots, save=save_plots)
            if additional_members_plot:
                plot_sources(member_candidates, save_file=os.path.join(cluster_dir, 'additional_members.png'),
                             colour=cluster.isochrone_colour, comparison=comparison, field_sources=field_sources,
                             plot_type='unique_members', members_label=new_members_label,
                             title=f'{cluster.name}'.replace('_', ' '), limits=limits, show_isochrone=True,
                             show_boundaries=True, cluster=cluster, show=show_plots, save=save_plots)
            if missed_members_plot:
                plot_sources(comparison, save_file=os.path.join(cluster_dir, 'missed_members.png'),
                             colour=cluster.isochrone_colour, comparison=member_candidates, field_sources=field_sources,
                             plot_type='unique_members', members_label=cluster_comparison_label,
                             title=f'{cluster.name}'.replace('_', ' '), limits=limits, show_isochrone=True,
                             show_boundaries=True, cluster=cluster, show=show_plots, save=save_plots)
        print(f'done, saved in {os.path.abspath(cluster_dir)}')
        print(' ')
        print(100 * '=')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--clusters_dir', nargs='?', type=str, default='clusters',
                        help='Directory where cluster data and results are saved.')
    parser.add_argument('--model_dir', nargs='?', type=str, default='deep_sets_model',
                        help= 'Directory of the model, which contains its (hyper)parameters, that will be used'
                              'to evaluate the candidate sources.')
    parser.add_argument('--n_samples', nargs='?', type=int, default=100,
                        help='Number of candidate samples to use for calculating the membership probability.')
    parser.add_argument('--size_support_set', nargs='?', type=int, default=10,
                        help='The number of members in the support set.')
    parser.add_argument('--hard_size_ss', nargs='?', type=bool, default=True,
                        help='When false, set the support set size to the number of available training members when '
                             'the former is larger than the latter.')
    parser.add_argument('--new_members_label', nargs='?', type=str, default='this study',
                        help='Label to use for indicating the training members in plots.')
    parser.add_argument('--candidate_prob_threshold', nargs='?', type=float, default=0.1,
                        help='Minimum membership probability of candidates plotted in the comparison and '
                             'additional members plot.')
    parser.add_argument('--new_members_prob_threshold', nargs='?', type=float, default=0.8,
                        help='Minimum membership probability of candidates plotted in the new members plot.')
    parser.add_argument('--use_tidal_radius', nargs='?', type=bool, default=False,
                        help='Whether to calculate the tidal radius of the cluster with the candidates'
                             'and exclude candidates outside the tidal radius from the plots.')
    parser.add_argument('--use_comparison_tidal_radius', nargs='?', type=bool, default=True,
                        help='Whether to calculate the tidal radius of the cluster '
                             'and exclude candidate members outside the tidal radius from the plots.')
    parser.add_argument('--new_members_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the candidates above a threshold probability.')
    parser.add_argument('--comparison_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the candidates and comparison members above a '
                             'threshold probability.')
    parser.add_argument('--additional_members_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the candidates above a threshold probability,'
                             'which are not present in the comparison members.')
    parser.add_argument('--missed_members_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the comparison members, which are not present'
                             'among the candidates above a threshold probability.')
    parser.add_argument('--venn_diagram_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing a Venn diagram comparing the candidates and the'
                             'comparison members above 10%, 50% and 90% membership probability.')
    parser.add_argument('--confusion_matrix_plot', nargs='?', type=bool, default=True,
                        help="Whether to create a plot showing how the candidate probability compares against"
                             "the probability given in the comparison. A 'confusion_matrix' is created by using "
                             "probability bins.")
    parser.add_argument('--density_profile_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the density profile of the candidates above a'
                             'threshold probability and the comparison members.')
    parser.add_argument('--mass_segregation_plot', nargs='?', type=bool, default=True,
                        help='Whether to create a plot showing the mass segregation of the candidates above a'
                             'threshold probability and the comparison members.')
    parser.add_argument('--plot_stellar_field', nargs='?', type=bool, default=True,
                        help='Whether to add the stellar field to the sources plots.')
    parser.add_argument('--plot_only', nargs='?', type=bool, default=False,
                        help='Whether to only create plots for the given clusters. This skips the (re)calculation of'
                             'candidate member probabilities.')
    parser.add_argument('--show_plots', nargs='?', type=bool, default=False,
                        help='Whether to show the plot.')
    parser.add_argument('--save_plots', nargs='?', type=bool, default=True,
                        help='Whether to save the plots.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the sampling of sources when determining membership '
                             'probabilities.')

    args_dict = vars(parser.parse_args())

    evaluate_clusters(args_dict['cluster_names'],
                      clusters_dir=args_dict['clusters_dir'],
                      model_dir=args_dict['model_dir'],
                      n_samples=args_dict['n_samples'],
                      size_support_set=args_dict['size_support_set'],
                      hard_size_ss=args_dict['hard_size_ss'],
                      new_members_label=args_dict['new_members_label'],
                      candidate_prob_threshold=args_dict['candidate_prob_threshold'],
                      new_members_prob_threshold=args_dict['member_prob_threshold'],
                      use_tidal_radius=args_dict['use_tidal_radius'],
                      use_comparison_tidal_radius=args_dict['use_comparison_tidal_radius'],
                      new_members_plot=args_dict['new_members_plot'],
                      comparison_plot=args_dict['comparison_plot'],
                      additional_members_plot=args_dict['additional_members_plot'],
                      missed_members_plot=args_dict['missed_members_plot'],
                      venn_diagram_plot=args_dict['venn_diagram_plot'],
                      confusion_matrix_plot=args_dict['confusion_matrix_plot'],
                      density_profile_plot=args_dict['density_profile_plot'],
                      mass_segregation_plot=args_dict['mass_segregation_plot'],
                      plot_stellar_field=args_dict['plot_stellar_field'],
                      plot_only=args_dict['plot_only'],
                      show_plots=args_dict['show'],
                      save_plots=args_dict['save_plots'],
                      seed=args_dict['seed'])
