import os
import argparse

from gaia_oc_amd.data_preparation.isochrone import make_isochrone
from gaia_oc_amd.data_preparation.parse_cone import parse_cone
from gaia_oc_amd.data_preparation.cones_query import download_cones
from gaia_oc_amd.data_preparation.sets import Sources
from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.utils import save_sets, load_cone, load_members, load_cluster_parameters, find_path
from gaia_oc_amd.candidate_evaluation.visualization import plot_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a string or a file with cluster names.')
    parser.add_argument('data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')
    parser.add_argument('isochrone_path', nargs='?', type=str, default='isochrones',
                        help='Path for retrieving isochrone data. Expects either a .dat file with isochrone data '
                             'or a directory with files formatted like "isochrones6.dat" '
                             'containing isochrone data of log(age) betweeen 6 and 7.')
    parser.add_argument('credentials_path', nargs='?', type=str, default='gaia_credentials',
                        help='Path to file containing a username and password for logging into the ESA Gaia archive.')
    parser.add_argument('cluster_path', nargs='?', type=str, default='cluster_parameters.tsv',
                        help='Path to file containing cluster properties, '
                             'i.e. 5d astrometric properties + errors + age + distance + extinction coefficient.')
    parser.add_argument('members_path', nargs='?', type=str, default='cg18_members.tsv',
                        help='Path to file containing open cluster members. '
                             'Required fields are the source identity, membership probability'
                             'and the cluster to which they belong.')
    parser.add_argument('comparison_path', nargs='?', type=str, default='t22_members.tsv',
                        help='Path to file containing open cluster members, which are to be compared against. '
                             'Same requirements as for the members_path.')
    parser.add_argument('cone_radius', nargs='?', type=float, default=60.,
                        help='Projected radius of the cone search.')
    parser.add_argument('cone_pm_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in proper motion '
                             'to look for sources for the cone search.')
    parser.add_argument('cone_plx_sigmas', nargs='?', type=float, default=10.,
                        help='How many sigmas away from the cluster mean in parallax '
                             'to look for sources for the cone search.')
    parser.add_argument('prob_threshold', nargs='?', type=float, default=1.0,
                        help='Minimum threshold for the probability of members to be used for training the model.')
    parser.add_argument('n_min_members', nargs='?', type=int, default=15,
                        help='Minimum number of members per cluster to use for training.')
    parser.add_argument('can_max_r', nargs='?', type=float, default=60.,
                        help='Maximum deviation in parallax (derived from radius in parsec) from the cluster mean '
                             'to be used in candidate selection.')
    parser.add_argument('can_pm_errors', nargs='?', type=float, default=5.,
                        help='Maximum deviation in proper motion (in number of sigma) from the cluster mean '
                             'to be used in candidate selection.')
    parser.add_argument('can_g_delta', nargs='?', type=float, default=1.5,
                        help='Maximum deviation in G magnitude from the isochrone '
                             'to be used in candidate selection.')
    parser.add_argument('can_bp_rp_delta', nargs='?', type=float, default=0.5,
                        help='Maximum deviation in colour from the isochrone '
                             'to be used in candidate selection.')
    parser.add_argument('can_source_errors', nargs='?', type=float, default=3.,
                        help='How many sigma candidates may lie outside the maximum deviations.')
    parser.add_argument('members_cluster_column', nargs='?', type=str, default='Cluster',
                        help='Column name for indicating the cluster of train member sources.')
    parser.add_argument('members_id_column', nargs='?', type=str, default='Source',
                        help='Column name for indicating the identity of train member sources.')
    parser.add_argument('members_prob_column', nargs='?', type=str, default='PMemb',
                        help='Column name for indicating the membership probability of train member sources.')
    parser.add_argument('comparison_cluster_column', nargs='?', type=str, default='Cluster',
                        help='Column name for indicating the cluster of comparison member sources.')
    parser.add_argument('comparison_id_column', nargs='?', type=str, default='GaiaEDR3',
                        help='Column name for indicating the identity of comparison member sources.')
    parser.add_argument('comparison_prob_column', nargs='?', type=str, default='Proba',
                        help='Column name for indicating the membership probability of comparison member sources.')
    parser.add_argument('members_label', nargs='?', type=str, default='train',
                        help='Label to use for indicating the training members in plots.')
    parser.add_argument('comparison_label', nargs='?', type=str, default='comparison',
                        help='Label to use for indicating the comparison members in plots.')
    parser.add_argument('show', nargs='?', type=bool, default=True,
                        help='Whether to show the candidates plot.')
    parser.add_argument('show_features', nargs='?', type=bool, default=False,
                        help='Whether to display feature arrows in the candidates plot.')
    parser.add_argument('show_boundaries', nargs='?', type=bool, default=True,
                        help='Whether to display zero-error boundaries in the candidates plot.')
    parser.add_argument('save_plot', nargs='?', type=bool, default=True,
                        help='Whether to save the candidates plot.')
    parser.add_argument('save_sets', nargs='?', type=bool, default=True,
                        help='Whether to save the source sets.')

    args = parser.parse_args()

    args_dict = vars(args)

    # cluster_names = [args_dict['cluster_names']]
    cluster_names = ['NGC_752']

    data_dir = args_dict['data_dir']
    data_dir = os.path.join(os.getcwd(), data_dir)
    isochrone_dir = find_path(args_dict['isochrone_path'], data_dir)
    credentials_path = find_path(args_dict['credentials_path'], data_dir)
    cluster_path = find_path(args_dict['cluster_path'], data_dir)
    members_path = find_path(args_dict['members_path'], data_dir)
    comparison_path = find_path(args_dict['comparison_path'], data_dir)

    cone_radius = args_dict['cone_radius']
    cone_pm_sigmas = args_dict['cone_pm_sigmas']
    cone_plx_sigmas = args_dict['cone_plx_sigmas']

    prob_threshold = args_dict['prob_threshold']
    n_min_members = args_dict['n_min_members']
    max_r = args_dict['can_max_r']
    pm_errors = args_dict['can_pm_errors']
    g_delta = args_dict['can_g_delta']
    bp_rp_delta = args_dict['can_bp_rp_delta']
    source_errors = args_dict['can_source_errors']

    members_cluster_column = args_dict['members_cluster_column']
    members_id_column = args_dict['members_id_column']
    members_prob_column = args_dict['members_prob_column']
    comparison_cluster_column = args_dict['comparison_cluster_column']
    comparison_id_column = args_dict['comparison_id_column']
    comparison_prob_column = args_dict['comparison_prob_column']

    members_label = args_dict['members_label']
    comparison_label = args_dict['comparison_label']

    show = args_dict['show']
    show_features = args_dict['show_features']
    show_boundaries = args_dict['show_boundaries']
    save_plot = args_dict['save_plot']
    save_set = args_dict['save_sets']

    print('Building sets for:', cluster_names)
    print('Number of clusters:', len(cluster_names))

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    for cluster_name in cluster_names:
        print(' ')
        print('Cluster:', cluster_name)
        print('Loading cluster paramters...', end=' ')
        cluster_params = load_cluster_parameters(cluster_name, cluster_path)
        if cluster_params is not None:
            print('done')
            cluster = Cluster(cluster_params)

            save_dir = os.path.join(data_dir, 'clusters', cluster.name)
            cone_path = os.path.join(save_dir, 'cone.vot.gz')
            if not os.path.exists(cone_path):
                download_cones([cluster], data_dir, credentials_path, cone_radius=cone_radius, pm_sigmas=cone_pm_sigmas,
                               plx_sigmas=cone_plx_sigmas)

            cone = load_cone(cone_path, cluster)

            print('Loading member data...', end=' ')
            if os.path.exists(members_path):
                members = load_members(members_path, cluster.name, cluster_column=members_cluster_column,
                                       id_column=members_id_column, prob_column=members_prob_column)
                hp_members = members[members['PMemb'] >= prob_threshold]
                while len(hp_members) < n_min_members and prob_threshold >= 0.0:
                    prob_threshold -= 0.1
                    hp_members = members[members['PMemb'] >= prob_threshold]
                if len(hp_members) >= n_min_members:
                    member_ids = hp_members['source_id']
                    member_probs = hp_members['PMemb']
                else:
                    member_ids, member_probs = None, None
            else:
                member_ids, member_probs = None, None

            if member_ids is not None:
                if os.path.exists(comparison_path):
                    comparison = load_members(comparison_path, cluster.name, cluster_column=comparison_cluster_column,
                                              id_column=comparison_id_column, prob_column=comparison_prob_column)
                    if len(comparison) > 0:
                        comparison_ids = comparison['source_id']
                        comparison_probs = comparison['PMemb']
                    else:
                        comparison_ids, comparison_probs = None, None
                else:
                    comparison_ids, comparison_probs = None, None
                print(f'done')

                isochrone = make_isochrone(isochrone_dir, cluster)

                members, candidates, non_members, comparison = parse_cone(cluster, cone, isochrone, member_ids,
                                                                          member_probs=member_probs,
                                                                          comparison_ids=comparison_ids,
                                                                          comparison_probs=comparison_probs,
                                                                          max_r=max_r, pm_errors=pm_errors,
                                                                          g_delta=g_delta, bp_rp_delta=bp_rp_delta,
                                                                          source_errors=source_errors)

                print(' ')
                print('Members:', len(members))
                print('Candidates:', len(candidates))
                print('Non-members:', len(non_members))
                print(' ')

                print('Plotting candidates...', end=' ')
                sources = Sources(members, candidates, non_members, comparison_members=comparison,
                                  members_label=members_label, comparison_label=comparison_label)
                plot_sources(sources, cluster, save_dir, isochrone=isochrone, plot_type='candidates',
                             show_boundaries=show_boundaries, show_features=show_features, show=show, save=save_plot)
                print('done')

                if save_set:
                    save_sets(data_dir, cluster, members, candidates, non_members, comparison)
            else:
                print(f'No members available! Skipping cluster {cluster.name}')
        else:
            print(f'No parameters available! Skipping cluster {cluster_name}')
