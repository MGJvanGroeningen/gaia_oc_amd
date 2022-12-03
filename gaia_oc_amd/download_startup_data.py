import os
import argparse

from gaia_oc_amd.data_preparation.query import query_vizier_catalog, query_isochrone


def download_startup_data(save_dir='./data'):
    """Downloads cluster parameters, isochrones and membership lists from a number of sources. Cluster parameters are
    taken from Cantat-Gaudin et al. (2020), isochrones are taken from http://stev.oapd.inaf.it/cgi-bin/cmd and
    we take membership lists from 4 papers: Cantat-Gaudin et al. (2018), Cantat-Gaudin & Anders (2020),
    Cantat-Gaudin et al. (2020) and Tarricq et al. (2022).

    Args:
        save_dir (str): Directory to save the data in.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Downloading cluster parameters...')

    cluster_params_catalog = 'J/A+A/640/A1/table1'
    cluster_params_columns = ['Cluster', 'RA_ICRS', 'DE_ICRS', 'pmRA*', 'pmDE', 'e_pmRA*', 'e_pmDE', 'plx', 'e_plx',
                              'AgeNN', 'AVNN', 'DistPc']
    cluster_params_save_path = os.path.join(save_dir, 'cluster_parameters.csv')

    query_vizier_catalog(cluster_params_catalog, cluster_params_save_path, cluster_params_columns,
                         new_column_names={'Cluster': 'name', 'RA_ICRS': 'ra', 'DE_ICRS': 'dec', 'pmRA_': 'pmra',
                                           'pmDE': 'pmdec', 'e_pmRA_': 'pmra_error', 'e_pmDE': 'pmdec_error',
                                           'plx': 'parallax', 'e_plx': 'parallax_error', 'AgeNN': 'age', 'AVNN': 'a0',
                                           'DistPc': 'dist'})
    print(f'Downloaded cluster parameters from Cantat-Gaudin et al. (2020), saved data at '
          f'{os.path.abspath(cluster_params_save_path)}')

    print(' ')
    print('Downloading isochrones...')
    isochrones_save_path = os.path.join(save_dir, 'isochrones.dat')
    query_isochrone(isochrones_save_path)
    print(f'Downloaded 400 isochrones from log(age)=6 to log(age)=9.99 with Z=0.0152, saved data at '
          f"{os.path.abspath(os.path.join(save_dir, 'isochrones.dat'))}.")

    print(' ')
    print('Downloading membership lists...')
    membership_lists = []

    cg18_members = {'catalog': 'J/A+A/618/A93/members',
                    'cluster_column': 'Cluster',
                    'source_id_column': 'Source',
                    'member_prob_column': 'PMemb',
                    'save_path': os.path.join(save_dir, 'cg18_members.csv'),
                    'label': 'Cantat-Gaudin et al. (2018)'}
    membership_lists.append(cg18_members)

    cg20a_members = {'catalog': 'J/A+A/633/A99/members',
                     'cluster_column': 'Cluster',
                     'source_id_column': 'Source',
                     'member_prob_column': 'Proba',
                     'save_path': os.path.join(save_dir, 'cg20a_members.csv'),
                     'label': 'Cantat-Gaudin & Anders (2020)'}
    membership_lists.append(cg20a_members)

    cg20b_members = {'catalog': 'J/A+A/640/A1/nodup',
                     'cluster_column': 'Cluster',
                     'source_id_column': 'GaiaDR2',
                     'member_prob_column': 'proba',
                     'save_path': os.path.join(save_dir, 'cg20b_members.csv'),
                     'label': 'Cantat-Gaudin et al. (2020)'}
    membership_lists.append(cg20b_members)

    t22_members = {'catalog': 'J/A+A/659/A59/table2',
                   'cluster_column': 'Cluster',
                   'source_id_column': 'GaiaEDR3',
                   'member_prob_column': 'Proba',
                   'save_path': os.path.join(save_dir, 't22_members.csv'),
                   'label': 'Tarricq et al. (2022)'}
    membership_lists.append(t22_members)

    for membership_list in membership_lists:
        query_vizier_catalog(membership_list['catalog'],
                             membership_list['save_path'],
                             [membership_list['cluster_column'],
                              membership_list['source_id_column'],
                              membership_list['member_prob_column']],
                             new_column_names={membership_list['cluster_column']: 'cluster',
                                               membership_list['source_id_column']: 'source_id',
                                               membership_list['member_prob_column']: 'PMemb'})
        print(f"Downloaded membership lists from {membership_list['label']}, saved data at "
              f"{os.path.abspath(membership_list['save_path'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', nargs='?', type=str, default='./data',
                        help='Directory where member and cluster data will be saved.')
    args_dict = vars(parser.parse_args())

    download_startup_data(args_dict['save_dir'])
