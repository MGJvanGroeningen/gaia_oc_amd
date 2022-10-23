import os
import time
import numpy as np
import argparse


from gaia_oc_amd.io import cluster_list, load_cluster_parameters, load_cone, load_members, load_isochrone, load_sets, \
    save_sets, save_cluster

from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.query import cone_search
from gaia_oc_amd.data_preparation.cone_preprocessing import cone_preprocessing
from gaia_oc_amd.data_preparation.source_sets import member_set, candidate_and_non_member_set
from gaia_oc_amd.data_preparation.features import add_features

from gaia_oc_amd.candidate_evaluation.visualization import plot_sources, plot_sources_limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--clusters_dir', nargs='?', type=str, default='clusters',
                        help='Directory where cluster data (e.g. cone searches, source sets) '
                             'and results will be saved.')
    parser.add_argument('--isochrone_path', nargs='?', type=str, default='data/isochrones.dat',
                        help='Path to file with isochrone data. Expects a .dat file with isochrones '
                             'between log(age) of 6 and 10.')
    parser.add_argument('--credentials_path', nargs='?', type=str, default='gaia_credentials',
                        help='Path to file containing a username and password for logging into the ESA Gaia archive.')
    parser.add_argument('--cluster_path', nargs='?', type=str, default='data/cluster_parameters.csv',
                        help='Path to file containing cluster properties, '
                             'i.e. 5d astrometric properties + errors + age + distance + extinction coefficient.')
    parser.add_argument('--members_path', nargs='?', type=str, default='data/cg20_members.csv',
                        help='Path to file containing open cluster members. '
                             'Required fields are the source identity, membership probability'
                             'and the cluster to which they belong.')
    parser.add_argument('--comparison_path', nargs='?', type=str, default='data/t22_members.csv',
                        help='Path to file containing open cluster members, which are used for comparison. '
                             'Same requirements as for the members_path.')
    parser.add_argument('--members_label', nargs='?', type=str, default='CG20',
                        help='Label for indicating the training members.')
    parser.add_argument('--comparison_label', nargs='?', type=str, default='T22',
                        help='Label for indicating the comparison members.')
    parser.add_argument('--cone_radius', nargs='?', type=float, default=50.,
                        help='Projected radius of the cone search.')
    parser.add_argument('--cone_pm_sigmas', nargs='?', type=float, default=7.,
                        help='How many sigmas away from the cluster mean in proper motion '
                             'to look for sources for the cone search.')
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
    parser.add_argument('--can_pm_error_weight', nargs='?', type=float, default=3.,
                        help='Maximum deviation in proper motion (in number of sigma) from the cluster mean '
                             'to be used in candidate selection.')
    parser.add_argument('--can_r_max_margin', nargs='?', type=float, default=15.,
                        help='Margin added to the maximum radius used for the parallax delta.')
    parser.add_argument('--can_c_margin', nargs='?', type=float, default=0.25,
                        help='Margin added to colour delta.')
    parser.add_argument('--can_g_margin', nargs='?', type=float, default=0.75,
                        help='Margin added to magnitude delta.')
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
    parser.add_argument('--overwrite_cone', nargs='?', type=bool, default=False,
                        help='Whether to overwrite the current cone votable if it already exists.')

    args_dict = vars(parser.parse_args())

    # main arguments
    clusters_dir = args_dict['clusters_dir']
    cluster_names = cluster_list(args_dict['cluster_names'])

    # path arguments
    isochrone_path = args_dict['isochrone_path']
    credentials_path = args_dict['credentials_path']
    cluster_path = args_dict['cluster_path']
    members_path = args_dict['members_path']
    comparison_path = args_dict['comparison_path']

    # labels
    members_label = args_dict['members_label']
    comparison_label = args_dict['comparison_label']

    # cone search arguments
    cone_radius = args_dict['cone_radius']
    cone_pm_sigmas = args_dict['cone_pm_sigmas']
    cone_plx_sigmas = args_dict['cone_plx_sigmas']
    overwrite_cone = args_dict['overwrite_cone']

    # training members arguments
    prob_threshold = args_dict['prob_threshold']
    n_min_members = args_dict['n_min_members']
    bootstrap_members = args_dict['bootstrap_members']
    bootstrap_prob = args_dict['bootstrap_probability']

    # candidate selection arguments
    pm_error_weight = args_dict['can_pm_error_weight']
    r_max_margin = args_dict['can_r_max_margin']
    c_margin = args_dict['can_c_margin']
    g_margin = args_dict['can_g_margin']
    source_error_weight = args_dict['can_source_error_weight']

    # plot arguments
    show = args_dict['show']

    # save arguments
    save_plot = args_dict['save_plot']
    save_source_sets = args_dict['save_source_sets']
    save_cluster_params = args_dict['save_cluster_params']

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
        if os.path.exists(cluster_path):
            cluster_params = load_cluster_parameters(cluster_path, cluster_name)
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

                print('Loading cone data...', end=' ')
                cone = load_cone(cone_path)
                cone = cone_preprocessing(cone, cluster.a0)
                print(f'done')

                print('Loading member data...', end=' ')
                if os.path.exists(members_path):
                    if bootstrap_members:
                        _, candidates, _, _ = load_sets(cluster_dir)
                        members = candidates
                        hp_members = members[members['PMemb'] >= bootstrap_prob]
                    else:
                        members = load_members(members_path, cluster.name)
                        hp_members = members[members['PMemb'] >= prob_threshold]

                    if len(hp_members) >= n_min_members:
                        member_ids = hp_members['source_id']
                        member_probs = hp_members['PMemb']
                        cluster.training_members_label = members_label
                        print(f'done')

                        print('Loading comparison member data...', end=' ')
                        if os.path.exists(comparison_path):
                            comparison = load_members(comparison_path, cluster.name)
                            if len(comparison) > 0:
                                comparison_ids = comparison['source_id']
                                comparison_probs = comparison['PMemb']
                                cluster.comparison_members_label = comparison_label
                                print('done')
                            else:
                                comparison_ids = members['source_id']
                                comparison_probs = members['PMemb']
                                cluster.comparison_members_label = members_label
                                print(f'No comparison members available for cluster {cluster.name} '
                                      f'at {os.path.abspath(comparison_path)}, using training members as comparison')
                        else:
                            comparison_ids, comparison_probs = None, None
                            print(f'No comparison members for cluster {cluster.name} were loaded, path to '
                                  f'comparison members {os.path.abspath(comparison_path)} does not exist.')

                        print('Parsing cone... ', end=' ')
                        # Construct the member set
                        train_members = member_set(cone, member_ids, member_probs)

                        # Update the cluster parameters based on the members
                        cluster.update_astrometric_parameters(train_members)

                        # Load the isochrone
                        isochrone = load_isochrone(isochrone_path, cluster.age, cluster.dist)

                        # Set cluster parameters that are relevant for the candidate selection and training features
                        cluster.set_candidate_selection_parameters(train_members, isochrone,
                                                                   pm_error_weight=pm_error_weight,
                                                                   r_max_margin=r_max_margin, c_margin=c_margin,
                                                                   g_margin=g_margin,
                                                                   source_error_weight=source_error_weight)

                        # Construct the candidate and non-member set
                        candidates, non_members = candidate_and_non_member_set(cone, cluster)

                        # Optionally construct the comparison set
                        if comparison_ids is not None:
                            comparison_members = member_set(cone, comparison_ids, comparison_probs)
                        else:
                            comparison_members = None

                        # Add the custom training features to the columns of the source dataframes
                        add_features([train_members, candidates, non_members, comparison_members], cluster)
                        print(f'done')

                        print(' ')
                        print('Members:', len(train_members))
                        print('Candidates:', len(candidates))
                        print('Non-members:', len(non_members))
                        print(' ')

                        print('Plotting candidates...', end=' ')
                        plot_sources(train_members, save_file=os.path.join(cluster_dir, 'candidates.png'),
                                     candidates=candidates, field_sources=non_members, plot_type='candidates',
                                     title=f'{cluster.name}'.replace('_', ' '), limits=plot_sources_limits(cone),
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
        print(100 * '=')
