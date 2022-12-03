import os
import time
import numpy as np
import pandas as pd
import argparse

from gaia_oc_amd.io import cluster_list, load_cluster_parameters, load_cone, load_members, load_isochrone, load_sets, \
    save_sets, save_cluster

from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.query import cone_search
from gaia_oc_amd.data_preparation.cone_preprocessing import cone_preprocessing
from gaia_oc_amd.data_preparation.isochrone_preprocessing import isochrone_preprocessing
from gaia_oc_amd.data_preparation.source_sets import member_set, candidate_and_non_member_set, get_duplicate_sources
from gaia_oc_amd.data_preparation.features import add_features

from gaia_oc_amd.candidate_evaluation.visualization import plot_sources, plot_sources_limits


def build_sets(cluster_names, clusters_dir='./data/clusters', cluster_parameters_path='./data/cluster_parameters.csv',
               train_members_path='./data/cg20b_members.csv', credentials_path='./gaia_credentials',
               isochrone_path='./data/isochrones.dat', comparison_members_path='./data/t22_members.csv',
               train_members_label='CG20', comparison_members_label='T22', cone_radius=50., cone_pm_sigmas=10.,
               cone_plx_sigmas=10., overwrite_cone=False, prob_threshold=1.0, n_min_members=15, bootstrap_members=False,
               bootstrap_prob=0.96, colour='g_rp', pm_error_weight=3.0, r_max_margin=15., c_margin=0.1, g_margin=0.8,
               source_error_weight=3.0, show=False, save_plot=True, save_source_sets=True, save_cluster_params=True,
               fast_mode=False):
    """Main function for preparing the datasets that can be used for training a model and evaluating candidate members.
    This includes the following steps:
        - Download source data through cone searches for the supplied clusters (use the 'overwrite_cone' keyword to
            overwrite already downloaded data).
        - Construct member dataframes by cross-matching source ids of supplied membership lists with the cone sources.
        - Label cone sources as either candidate members or non-members through a candidate selection process.
        - Optionally plot the members, candidates and non-members and save their data.

    Args:
        cluster_names (str, list): 'Names of the open cluster(s) we want to build sets for. Can be a name or a file
            with cluster names.'
        clusters_dir (str): 'Directory where cluster data (e.g. cone searches, source sets) and results will be saved.'
        cluster_parameters_path (str): 'Path to file containing cluster properties, i.e. 5d astrometric properties +
            astrometric errors + age + distance + extinction coefficient.'
        train_members_path (str): 'Path to file containing open cluster members. Required fields are the source
            identity, membership probability and the cluster to which they belong.'
        credentials_path (str): 'Path to file containing a username and password for logging into the ESA Gaia archive.'
        isochrone_path (str): 'Path to file with isochrone data. Expects a .dat file with isochrones between log(age)
            of 6 and 10.'
        comparison_members_path (str): 'Path to file containing open cluster members, which are used for comparison.
            Same requirements as for the members_path.'
        train_members_label (str): 'Label for indicating the training members.'
        comparison_members_label (str): 'Label for indicating the comparison members.'
        cone_radius (float): 'Projected radius of the cone search.'
        cone_pm_sigmas (float): 'How many sigmas away from the cluster mean in proper motion to include sources in
            the cone search.'
        cone_plx_sigmas (float): 'How many sigmas away from the cluster mean in parallax to look for sources for the
            cone search.'
        overwrite_cone (bool): 'Whether to overwrite the current cone votable if it already exists.'
        prob_threshold (float): 'Minimum threshold for the probability of members to be used for training the model.
            This threshold is exceeded if there are less than n_min_members members'
        n_min_members (int): 'Minimum number of members per cluster to use for training. This must be greater than the
            number of members used for the support set (5 by default).'
        bootstrap_members (bool): 'Whether to use the candidates above a certain probability as the training members.'
        bootstrap_prob (float): 'The minimum candidate probability when bootstrapping members.'
        colour (str): "Colour to use for the isochrone condition. ('bp_rp', 'g_rp')"
        pm_error_weight (float): 'Maximum deviation in proper motion (in number of sigma) from the cluster mean to be
            used in candidate selection.'
        r_max_margin (float): 'Margin added to the maximum radius used for the parallax delta.'
        c_margin (float): 'Margin added to colour delta.'
        g_margin (float): 'Margin added to magnitude delta.'
        source_error_weight (float): 'How many sigma candidates may lie outside the maximum separation deltas.'
        show (bool): 'Whether to show the candidates plot.'
        save_plot (bool): 'Whether to save the candidates plot.'
        save_source_sets (bool): 'Whether to save the source sets.'
        save_cluster_params (bool): 'Whether to save the cluster parameters.'
        fast_mode (bool): 'If True, use a (~4x) faster but more memory intensive method. Might run out of memory for '
                             'many (>~10^6) sources.'
    """

    cluster_names = cluster_list(cluster_names)
    n_clusters = len(cluster_names)
    print('Building sets for:', cluster_names)
    print('Number of clusters:', n_clusters)

    if not os.path.exists(clusters_dir):
        os.mkdir(clusters_dir)

    for idx, cluster_name in enumerate(cluster_names):
        t0 = time.time()
        print(' ')
        print('Cluster:', cluster_name, f' ({idx + 1} / {n_clusters})')
        cluster_dir = os.path.join(clusters_dir, cluster_name)
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)

        print('Loading cluster parameters...', end=' ')
        if os.path.exists(cluster_parameters_path):
            cluster_params = load_cluster_parameters(cluster_parameters_path, cluster_name)
            if cluster_params is not None:
                print('done')

                # Create a cluster object
                cluster = Cluster(cluster_params)
                if save_cluster_params and not os.path.exists(os.path.join(cluster_dir, 'cluster')):
                    save_cluster(cluster_dir, cluster)

                # Download sources with a cone search
                cone_path = os.path.join(cluster_dir, 'cone.vot.gz')
                if not os.path.exists(cone_path) or overwrite_cone:
                    cone_search(cluster, clusters_dir, credentials_path, cone_radius=cone_radius,
                                pm_sigmas=cone_pm_sigmas, plx_sigmas=cone_plx_sigmas)

                print('Preparing cone data...', end=' ')
                cone = load_cone(cone_path)
                cone = cone_preprocessing(cone)
                print(f'done')

                print('Creating member dataframes...', end=' ')
                if os.path.exists(train_members_path):
                    if bootstrap_members:
                        _, candidates, _, _ = load_sets(cluster_dir)
                        members = candidates
                        train_members = members[members['PMemb'] >= bootstrap_prob]
                    else:
                        members = load_members(train_members_path, cluster.name)
                        train_members = members[members['PMemb'] >= prob_threshold]

                    if len(train_members) >= n_min_members:
                        train_members_ids = train_members['source_id']
                        train_members_probs = train_members['PMemb']
                        cluster.set_train_members_label(train_members_label)

                        comparison_member_ids = members['source_id']
                        comparison_member_probs = members['PMemb']
                        cluster.set_comparison_members_label(train_members_label)

                        if os.path.exists(comparison_members_path):
                            comparison_members = load_members(comparison_members_path, cluster.name)
                            if len(comparison_members) > 0:
                                comparison_member_ids = comparison_members['source_id']
                                comparison_member_probs = comparison_members['PMemb']
                                cluster.comparison_members_label = comparison_members_label
                            else:
                                print(f'No comparison members available for cluster {cluster.name} '
                                      f'at {os.path.abspath(comparison_members_path)}, using training members as '
                                      f'comparison.')
                        else:
                            print(f'The comparison members path {os.path.abspath(comparison_members_path)} does not '
                                  f'exist, using training members as comparison.')
                        print(f'done')

                        print('Parsing cone... ', end=' ')
                        # Construct the member set
                        train_members = member_set(cone, train_members_ids, train_members_probs)

                        # Update the cluster parameters based on the members
                        cluster.update_astrometric_parameters(train_members)

                        # Load the isochrone
                        isochrone = load_isochrone(isochrone_path, cluster.age)
                        isochrone = isochrone_preprocessing(isochrone, cluster.dist, colour=colour, a0=cluster.a0)

                        # Set cluster parameters that are relevant for the candidate selection and training features
                        cluster.set_candidate_selection_parameters(train_members, isochrone,
                                                                   colour=colour,
                                                                   pm_error_weight=pm_error_weight,
                                                                   r_max_margin=r_max_margin, c_margin=c_margin,
                                                                   g_margin=g_margin,
                                                                   source_error_weight=source_error_weight)

                        # Construct the candidate and non-member set
                        candidates, non_members = candidate_and_non_member_set(cone, cluster)

                        # Label sources that are both training members and non-members as candidates
                        dubious_members = get_duplicate_sources(train_members, non_members, keep='last')
                        if len(dubious_members) > 0:
                            train_members = train_members[~train_members['source_id'].isin(
                                dubious_members['source_id'])].copy()
                            non_members = non_members[~non_members['source_id'].isin(
                                dubious_members['source_id'])].copy()
                            candidates = pd.concat((candidates, dubious_members))

                        # Construct the comparison set
                        comparison_members = member_set(cone, comparison_member_ids, comparison_member_probs)

                        # Add the custom training features to the columns of the source dataframes
                        add_features([train_members, candidates, non_members, comparison_members], cluster,
                                     fast_mode=fast_mode)
                        print(f'done')

                        print(' ')
                        print('Members:', len(train_members))
                        print('Candidates:', len(candidates))
                        print('Non-members:', len(non_members))
                        print(' ')

                        print('Plotting candidates...', end=' ')
                        plot_sources(train_members, save_file=os.path.join(cluster_dir, 'candidates.png'),
                                     colour=cluster.isochrone_colour,
                                     candidates=candidates, field_sources=non_members, plot_type='candidates',
                                     title=f'{cluster.name}'.replace('_', ' '),
                                     limits=plot_sources_limits(cone, cluster.isochrone_colour),
                                     show_isochrone=True, show_boundaries=True, cluster=cluster, show=show,
                                     save=save_plot)
                        print('done')

                        print('Saving source sets and cluster data...', end=' ')
                        if save_source_sets:
                            save_sets(cluster_dir, train_members, candidates, non_members,
                                      comparison_members)

                        if save_cluster_params:
                            save_cluster(cluster_dir, cluster)
                        print(f'done, saved in {os.path.abspath(cluster_dir)}')

                    else:
                        print(f'There are only {len(train_members)} members available available with a probability '
                              f'of {np.round(100 * prob_threshold, 1)}% or higher for cluster {cluster.name}, '
                              f'which is less than the minimum number of members ({n_min_members}), '
                              f'skipping cluster {cluster.name}')
                else:
                    raise ValueError(f'The path from which to load the training members '
                                     f'{os.path.abspath(train_members_path)} does not exist!')
            else:
                print(f'No cluster parameters available for cluster {cluster_name} at '
                      f'{os.path.abspath(cluster_parameters_path)}, skipping cluster {cluster_name}')
        else:
            raise ValueError(f'The path from which to load the cluster parameters '
                             f'{os.path.abspath(cluster_parameters_path)} does not exist!')
        print(f'Cluster processed in {np.round(time.time() - t0, 1)} sec')
        print(' ')
        print(100 * '=')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--clusters_dir', nargs='?', type=str, default='./clusters',
                        help='Directory where cluster data (e.g. cone searches, source sets) '
                             'and results will be saved.')
    parser.add_argument('--isochrone_path', nargs='?', type=str, default='./data/isochrones.dat',
                        help='Path to file with isochrone data. Expects a .dat file with isochrones '
                             'between log(age) of 6 and 10.')
    parser.add_argument('--credentials_path', nargs='?', type=str, default='./data/gaia_credentials',
                        help='Path to file containing a username and password for logging into the ESA Gaia archive.')
    parser.add_argument('--cluster_parameters_path', nargs='?', type=str, default='./data/cluster_parameters.csv',
                        help='Path to file containing cluster properties, '
                             'i.e. 5d astrometric properties + errors + age + distance + extinction coefficient.')
    parser.add_argument('--members_path', nargs='?', type=str, default='./data/cg20a_members.csv',
                        help='Path to file containing open cluster members. '
                             'Required fields are the source identity, membership probability'
                             'and the cluster to which they belong.')
    parser.add_argument('--comparison_path', nargs='?', type=str, default='./data/t22_members.csv',
                        help='Path to file containing open cluster members, which are used for comparison. '
                             'Same requirements as for the members_path.')
    parser.add_argument('--members_label', nargs='?', type=str, default='CG20',
                        help='Label for indicating the training members.')
    parser.add_argument('--comparison_label', nargs='?', type=str, default='T22',
                        help='Label for indicating the comparison members.')
    parser.add_argument('--cone_radius', nargs='?', type=float, default=50.,
                        help='Projected radius of the cone search.')
    parser.add_argument('--cone_pm_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in proper motion '
                             'to include sources for the cone search.')
    parser.add_argument('--cone_plx_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in parallax '
                             'to look for sources for the cone search.')
    parser.add_argument('--prob_threshold', nargs='?', type=float, default=1.0,
                        help='Minimum threshold for the probability of members to be used for training the model. '
                             'This threshold is exceeded if there are less than n_min_members members')
    parser.add_argument('--n_min_members', nargs='?', type=int, default=15,
                        help='Minimum number of members per cluster to use for training. This must be greater than'
                             'the number of members used for the support set (5 by default).')
    parser.add_argument('--bootstrap_members', nargs='?', type=bool, default=False,
                        help='Whether to use the candidates above a certain probability as the training members.')
    parser.add_argument('--bootstrap_probability', nargs='?', type=float, default=0.96,
                        help='The minimum candidate probability when bootstrapping members.')
    parser.add_argument('--can_colour', nargs='?', type=str, default='g_rp',
                        help="Colour to use for the isochrone condition. ('bp_rp', 'g_rp')")
    parser.add_argument('--can_pm_error_weight', nargs='?', type=float, default=3.,
                        help='Maximum deviation in proper motion (in number of sigma) from the cluster mean '
                             'to be used in candidate selection.')
    parser.add_argument('--can_r_max_margin', nargs='?', type=float, default=15.,
                        help='Margin added to the maximum radius used for the parallax delta.')
    parser.add_argument('--can_c_margin', nargs='?', type=float, default=0.1,
                        help='Margin added to colour delta.')
    parser.add_argument('--can_g_margin', nargs='?', type=float, default=0.8,
                        help='Margin added to magnitude delta.')
    parser.add_argument('--can_source_error_weight', nargs='?', type=float, default=3.,
                        help='How many sigma candidates may lie outside the maximum deviations.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the candidates plot.')
    parser.add_argument('--save_plot', nargs='?', type=bool, default=True,
                        help='Whether to save the candidates plot.')
    parser.add_argument('--save_source_sets', nargs='?', type=bool, default=True,
                        help='Whether to save the source sets.')
    parser.add_argument('--save_cluster_params', nargs='?', type=bool, default=True,
                        help='Whether to save the cluster parameters.')
    parser.add_argument('--overwrite_cone', nargs='?', type=bool, default=False,
                        help='Whether to overwrite the current cone votable if it already exists.')
    parser.add_argument('--fast_mode', nargs='?', type=bool, default=False,
                        help='If True, use a faster but more memory intensive method, which might crash for '
                             'many (>~10^6) sources.')

    args_dict = vars(parser.parse_args())

    build_sets(args_dict['cluster_names'],
               clusters_dir=args_dict['clusters_dir'],
               cluster_parameters_path=args_dict['cluster_parameters_path'],
               train_members_path=args_dict['train_members_path'],
               credentials_path=args_dict['credentials_path'],
               isochrone_path=args_dict['isochrone_path'],
               comparison_members_path=args_dict['comparison_members_path'],
               train_members_label=args_dict['train_members_label'],
               comparison_members_label=args_dict['comparison_members_label'],
               cone_radius=args_dict['cone_radius'],
               cone_pm_sigmas=args_dict['cone_pm_sigmas'],
               cone_plx_sigmas=args_dict['cone_plx_sigmas'],
               overwrite_cone=args_dict['overwrite_cone'],
               prob_threshold=args_dict['prob_threshold'],
               n_min_members=args_dict['n_min_members'],
               bootstrap_members=args_dict['bootstrap_members'],
               bootstrap_prob=args_dict['bootstrap_prob'],
               colour=args_dict['can_colour'],
               pm_error_weight=args_dict['can_pm_error_weight'],
               r_max_margin=args_dict['can_r_max_margin'],
               c_margin=args_dict['can_c_margin'],
               g_margin=args_dict['can_g_margin'],
               source_error_weight=args_dict['can_source_error_weight'],
               show=args_dict['show'],
               save_plot=args_dict['save_plot'],
               save_source_sets=args_dict['save_source_sets'],
               save_cluster_params=args_dict['save_cluster_params'],
               fast_mode=args_dict['fast_mode'])
