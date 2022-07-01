import os
import argparse

from gaia_oc_amd.data_preparation.query import query_catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')

    args_dict = vars(parser.parse_args())
    data_dir = args_dict['data_dir']
    if not os.path.exists(data_dir):
        default_data_dir = os.path.join(os.getcwd(), 'data')
        print(f'Supplied data path {data_dir} does not exist, using default path {default_data_dir}')
        if not os.path.exists(default_data_dir):
            os.mkdir(default_data_dir)
        data_dir = default_data_dir

    cg18_members_catalog = 'J/A+A/618/A93/members'
    t22_members_catalog = 'J/A+A/659/A59/table2'
    cluster_params_catalog = 'J/A+A/640/A1/table1'

    catalogs = [cg18_members_catalog, t22_members_catalog, cluster_params_catalog]

    cg18_members_columns = ['Cluster', 'Source', 'PMemb']
    t22_members_columns = ['Cluster', 'GaiaEDR3', 'Proba']
    cluster_params_columns = ['Cluster', 'RA_ICRS', 'DE_ICRS', 'pmRA*', 'pmDE', 'e_pmRA*', 'e_pmDE', 'plx', 'e_plx',
                              'AgeNN', 'AVNN', 'DistPc']

    columns = [cg18_members_columns, t22_members_columns, cluster_params_columns]

    cg18_members_save_path = os.path.join(data_dir, 'cg18_members.csv')
    t22_members_save_path = os.path.join(data_dir, 't22_members.csv')
    cluster_params_save_path = os.path.join(data_dir, 'cluster_parameters.csv')

    save_paths = [cg18_members_save_path, t22_members_save_path, cluster_params_save_path]
    labels = ['cg18 members', 't22 members', 'cluster parameters']

    print('Downloading members and cluster parameters...')
    for catalog, cols, save_path, label in zip(catalogs, columns, save_paths, labels):
        if label == 'cg18 members' or label == 't22 members':
            query_catalog(catalog, cols, save_path, cluster_column=cols[0], id_column=cols[1], prob_column=cols[2])
        else:
            query_catalog(catalog, cols, save_path)
        print(f'Downloaded {label}, saved data at {os.path.abspath(save_path)}')
