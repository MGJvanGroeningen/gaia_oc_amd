import os
import argparse

from gaia_oc_amd.data_preparation.query import query_isochrone

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')

    args_dict = vars(parser.parse_args())
    data_dir = args_dict['data_dir']
    if not os.path.exists(data_dir):
        default_data_dir = os.path.join(os.getcwd(), 'data')
        print(f'Using default save path {default_data_dir}')
        if not os.path.exists(default_data_dir):
            os.mkdir(default_data_dir)
        data_dir = default_data_dir

    query_isochrone(data_dir)
