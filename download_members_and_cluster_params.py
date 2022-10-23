import os
import argparse

from gaia_oc_amd.data_preparation.query import query_catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', nargs='?', type=str, default='data',
                        help='Directory where member and cluster data will be saved.')
    args_dict = vars(parser.parse_args())
    save_dir = args_dict['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Downloading members...')

    # cg18_members_catalog = 'J/A+A/618/A93/members'
    # cg20_2_members_catalog = 'J/A+A/640/A1/nodup'
    cg20_members_catalog = 'J/A+A/633/A99/members'
    t22_members_catalog = 'J/A+A/659/A59/table2'

    # cg18_members_columns = ['Cluster', 'Source', 'PMemb']
    # cg20_2_members_columns = ['Cluster', 'GaiaDR2', 'proba']
    cg20_members_columns = ['Cluster', 'Source', 'Proba']
    t22_members_columns = ['Cluster', 'GaiaEDR3', 'Proba']

    # cg18_members_save_path = 'data/cg18_members.csv'
    # cg20_2_members_save_path = 'data/cg20_2_members.csv'
    cg20_members_save_path = os.path.join(save_dir, 'cg20_members.csv')
    t22_members_save_path = os.path.join(save_dir, 't22_members.csv')

    catalogs = [cg20_members_catalog, t22_members_catalog]
    columns = [cg20_members_columns, t22_members_columns]
    save_paths = [cg20_members_save_path, t22_members_save_path]
    labels = ['cg20 members', 't22 members']

    for catalog, cols, save_path, label in zip(catalogs, columns, save_paths, labels):
        query_catalog(catalog, cols, save_path, cluster_column=cols[0], id_column=cols[1], prob_column=cols[2])
        print(f'Downloaded {label}, saved data at {os.path.abspath(save_path)}')

    print(' ')
    print('Downloading cluster parameters...')

    cluster_params_catalog = 'J/A+A/640/A1/table1'
    cluster_params_columns = ['Cluster', 'RA_ICRS', 'DE_ICRS', 'pmRA*', 'pmDE', 'e_pmRA*', 'e_pmDE', 'plx', 'e_plx',
                              'AgeNN', 'AVNN', 'DistPc']
    cluster_params_save_path = os.path.join(save_dir, 'cluster_parameters.csv')

    query_catalog(cluster_params_catalog, cluster_params_columns, cluster_params_save_path)
    print(f'Downloaded cluster parameters, saved data at {os.path.abspath(cluster_params_save_path)}')
