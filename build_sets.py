import os
import time
import numpy as np
import argparse

from gaia_oc_amd.data_preparation.parse_cone import parse_cone
from gaia_oc_amd.data_preparation.query import cone_search
from gaia_oc_amd.data_preparation.sets import Sources
from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.io import save_cluster, save_sets, load_cluster_parameters, load_cone, load_members, \
    load_isochrone, cluster_list, find_path
from gaia_oc_amd.candidate_evaluation.visualization import plot_sources


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
    parser.add_argument('--credentials_path', nargs='?', type=str, default='gaia_credentials',
                        help='Path to file containing a username and password for logging into the ESA Gaia archive.')
    parser.add_argument('--cluster_path', nargs='?', type=str, default='cluster_parameters.csv',
                        help='Path to file containing cluster properties, '
                             'i.e. 5d astrometric properties + errors + age + distance + extinction coefficient.')
    parser.add_argument('--members_path', nargs='?', type=str, default='cg18_members.csv',
                        help='Path to file containing open cluster members. '
                             'Required fields are the source identity, membership probability'
                             'and the cluster to which they belong.')
    parser.add_argument('--comparison_path', nargs='?', type=str, default='t22_members.csv',
                        help='Path to file containing open cluster members, which are used for comparison. '
                             'Same requirements as for the members_path.')
    parser.add_argument('--cone_radius', nargs='?', type=float, default=60.,
                        help='Projected radius of the cone search.')
    parser.add_argument('--cone_pm_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in proper motion '
                             'to look for sources for the cone search.')
    parser.add_argument('--cone_plx_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in parallax '
                             'to look for sources for the cone search.')
    parser.add_argument('--cone_verbose', nargs='?', type=bool, default=False,
                        help='Whether to print information about the cone search download process.')
    parser.add_argument('--prob_threshold', nargs='?', type=float, default=1.0,
                        help='Minimum threshold for the probability of members to be used for training the model. '
                             'This threshold is exceeded if there are less than n_min_members members')
    parser.add_argument('--n_min_members', nargs='?', type=int, default=15,
                        help='Minimum number of members per cluster to use for training. This must be greater than'
                             'the number of members used for the support set (5 by default).')
    parser.add_argument('--members_cluster_column', nargs='?', type=str, default='Cluster',
                        help='Column name for indicating the cluster of train member sources.')
    parser.add_argument('--members_id_column', nargs='?', type=str, default='Source',
                        help='Column name for indicating the identity of train member sources.')
    parser.add_argument('--members_prob_column', nargs='?', type=str, default='PMemb',
                        help='Column name for indicating the membership probability of train member sources.')
    parser.add_argument('--comparison_cluster_column', nargs='?', type=str, default='Cluster',
                        help='Column name for indicating the cluster of comparison member sources.')
    parser.add_argument('--comparison_id_column', nargs='?', type=str, default='GaiaEDR3',
                        help='Column name for indicating the identity of comparison member sources.')
    parser.add_argument('--comparison_prob_column', nargs='?', type=str, default='Proba',
                        help='Column name for indicating the membership probability of comparison member sources.')
    parser.add_argument('--can_max_r', nargs='?', type=float, default=60.,
                        help='Maximum radius in parsec which determines the maximum deviation in parallax '
                             'from the cluster mean to be used in candidate selection.')
    parser.add_argument('--can_pm_error_weight', nargs='?', type=float, default=5.,
                        help='Maximum deviation in proper motion (in number of sigma) from the cluster mean '
                             'to be used in candidate selection.')
    parser.add_argument('--can_g_delta', nargs='?', type=float, default=1.5,
                        help='Maximum deviation in G magnitude from the isochrone '
                             'to be used in candidate selection.')
    parser.add_argument('--can_bp_rp_delta', nargs='?', type=float, default=0.5,
                        help='Maximum deviation in colour from the isochrone '
                             'to be used in candidate selection.')
    parser.add_argument('--can_source_error_weight', nargs='?', type=float, default=3.,
                        help='How many sigma candidates may lie outside the maximum deviations.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the candidates plot.')
    parser.add_argument('--show_features', nargs='?', type=bool, default=False,
                        help='Whether to display feature arrows in the candidates plot.')
    parser.add_argument('--show_boundaries', nargs='?', type=bool, default=True,
                        help='Whether to display zero-error boundaries in the candidates plot.')
    parser.add_argument('--save_plot', nargs='?', type=bool, default=True,
                        help='Whether to save the candidates plot.')
    parser.add_argument('--save_source_sets', nargs='?', type=bool, default=True,
                        help='Whether to save the source sets.')
    parser.add_argument('--save_cluster_params', nargs='?', type=bool, default=True,
                        help='Whether to save the cluster parameters.')

    args_dict = vars(parser.parse_args())

    # main arguments
    data_dir = args_dict['data_dir']
    cluster_names = cluster_list(args_dict['cluster_names'], data_dir)

    # path arguments
    isochrone_path = find_path(args_dict['isochrone_path'], data_dir)
    credentials_path = find_path(args_dict['credentials_path'], data_dir)
    cluster_path = find_path(args_dict['cluster_path'], data_dir)
    members_path = find_path(args_dict['members_path'], data_dir)
    comparison_path = find_path(args_dict['comparison_path'], data_dir)

    # cone search arguments
    cone_radius = args_dict['cone_radius']
    cone_pm_sigmas = args_dict['cone_pm_sigmas']
    cone_plx_sigmas = args_dict['cone_plx_sigmas']
    cone_verbose = args_dict['cone_verbose']

    # members arguments
    prob_threshold = args_dict['prob_threshold']
    n_min_members = args_dict['n_min_members']
    members_cluster_column = args_dict['members_cluster_column']
    members_id_column = args_dict['members_id_column']
    members_prob_column = args_dict['members_prob_column']
    comparison_cluster_column = args_dict['comparison_cluster_column']
    comparison_id_column = args_dict['comparison_id_column']
    comparison_prob_column = args_dict['comparison_prob_column']

    # candidate selection arguments
    max_r = args_dict['can_max_r']
    pm_error_weight = args_dict['can_pm_error_weight']
    g_delta = args_dict['can_g_delta']
    bp_rp_delta = args_dict['can_bp_rp_delta']
    source_error_w = args_dict['can_source_error_weight']

    # plot arguments
    show = args_dict['show']
    show_features = args_dict['show_features']
    show_boundaries = args_dict['show_boundaries']

    # save arguments
    save_plot = args_dict['save_plot']
    save_source_sets = args_dict['save_source_sets']
    save_cluster_params = args_dict['save_cluster_params']

    n_clusters = len(cluster_names)
    print('Building sets for:', cluster_names)
    print('Number of clusters:', n_clusters)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for idx, cluster_name in enumerate(cluster_names):
        t0 = time.time()
        print(' ')
        print('Cluster:', cluster_name, f' ({idx + 1} / {n_clusters})')
        print('Loading cluster parameters...', end=' ')
        if os.path.exists(cluster_path):
            cluster_params = load_cluster_parameters(cluster_path, cluster_name)
            if cluster_params is not None:
                print('done')
                cluster = Cluster(cluster_params)

                cluster_dir = os.path.join(data_dir, 'clusters', cluster.name)
                cone_path = os.path.join(cluster_dir, 'cone.vot.gz')
                if not os.path.exists(cone_path):
                    cone_search(cluster, data_dir, credentials_path, cone_radius=cone_radius,
                                pm_sigmas=cone_pm_sigmas, plx_sigmas=cone_plx_sigmas, verbose=cone_verbose)

                print('Loading cone data...', end=' ')
                cone = load_cone(cone_path, cluster)
                print(f'done')

                print('Loading member data...', end=' ')
                if os.path.exists(members_path):
                    members = load_members(members_path, cluster.name, cluster_column=members_cluster_column,
                                           id_column=members_id_column, prob_column=members_prob_column)
                    hp_members = members[members['PMemb'] >= prob_threshold]
                    if len(hp_members) >= n_min_members:
                        member_ids = hp_members['source_id']
                        member_probs = hp_members['PMemb']
                        print(f'done')

                        print('Loading comparison member data...', end=' ')
                        if os.path.exists(comparison_path):
                            comparison = load_members(comparison_path, cluster.name,
                                                      cluster_column=comparison_cluster_column,
                                                      id_column=comparison_id_column,
                                                      prob_column=comparison_prob_column)
                            if len(comparison) > 0:
                                comparison_ids = comparison['source_id']
                                comparison_probs = comparison['PMemb']
                                print('done')
                            else:
                                comparison_ids, comparison_probs = None, None
                                print(f'No comparison members for cluster {cluster.name} were loaded, no '
                                      f'members available for cluster {cluster.name} '
                                      f'at {os.path.abspath(comparison_path)}')
                        else:
                            comparison_ids, comparison_probs = None, None
                            print(f'No comparison members for cluster {cluster.name} were loaded, path to '
                                  f'comparison members {os.path.abspath(comparison_path)} does not exist.')

                        isochrone = load_isochrone(isochrone_path, cluster)

                        print('Parsing cone... ', end=' ')
                        members, candidates, non_members, comparison = parse_cone(cone, cluster, isochrone, member_ids,
                                                                                  member_probs=member_probs,
                                                                                  comparison_ids=comparison_ids,
                                                                                  comparison_probs=comparison_probs,
                                                                                  max_r=max_r,
                                                                                  pm_error_weight=pm_error_weight,
                                                                                  g_delta=g_delta,
                                                                                  bp_rp_delta=bp_rp_delta,
                                                                                  source_error_weight=source_error_w)
                        print(f'done')

                        print(' ')
                        print('Members:', len(members))
                        print('Candidates:', len(candidates))
                        print('Non-members:', len(non_members))
                        print(' ')

                        print('Plotting candidates...', end=' ')
                        sources = Sources(members, candidates, non_members, comparison=comparison)
                        plot_sources(sources, cluster, cluster_dir, isochrone=isochrone, plot_type='candidates',
                                     show_boundaries=show_boundaries, show_features=show_features, show=show,
                                     save=save_plot)
                        print('done')

                        print('Saving source sets and cluster data...', end=' ')
                        if save_source_sets:
                            save_sets(cluster_dir, members, candidates, non_members, comparison)

                        if save_cluster_params:
                            save_cluster(cluster_dir, cluster)
                        print(f'done, saved in {os.path.abspath(cluster_dir)}')

                    else:
                        print(f'There are only {len(hp_members)} members available available with a probability '
                              f'of {np.round(100 * prob_threshold,1)}% or higher for cluster {cluster.name}, '
                              f'which is less than the minimum number of members ({n_min_members}), '
                              f'skipping cluster {cluster.name}')
                else:
                    raise ValueError(f'The path from which to load the members '
                                     f'{os.path.abspath(members_path)} does not exist!')
            else:
                print(f'No cluster parameters available for cluster {cluster_name} at '
                      f'{os.path.abspath(cluster_path)}, skipping cluster {cluster_name}')
        else:
            raise ValueError(f'The path from which to load the cluster parameters '
                             f'{os.path.abspath(cluster_path)} does not exist!')
        print(f'Cluster processed in {np.round(time.time() - t0, 1)} sec')
        print(' ')
        print(40 * '==')
