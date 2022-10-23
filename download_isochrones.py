import os
import argparse

from gaia_oc_amd.data_preparation.query import query_isochrone

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', nargs='?', type=str, default='data',
                        help='Directory where member and cluster data will be saved.')
    args_dict = vars(parser.parse_args())
    save_dir = args_dict['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    query_isochrone(save_dir)
