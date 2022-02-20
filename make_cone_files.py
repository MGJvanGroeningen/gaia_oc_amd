import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import os
import numpy as np
from filter_cone_data import get_cluster_parameters, fields


def download_cone_file(clusters, data_path):
    credentials_file = os.path.join(data_path, 'gaia_credentials')
    Gaia.login(credentials_file=credentials_file)

    # Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2, default
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"  # Select early Data Release 3
    Gaia.ROW_LIMIT = -1

    save_dir = os.path.join(data_path, 'cones')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_path = os.path.join(data_path, 'cluster_parameters.tsv')

    if type(clusters) is str:
        clusters = [clusters]

    cone_fields = fields['all']
    cone_fields.remove('PMemb')

    for cluster in clusters:
        if not os.path.exists(os.path.join(data_path, 'members', cluster + '.csv')):
            raise ValueError(f'The cluster {cluster} does not exist!')

        print(f'Downloading {cluster} cone...', end=' ')
        cluster_kwargs = get_cluster_parameters(param_path, cluster)
        coord = SkyCoord(ra=cluster_kwargs['ra'], dec=cluster_kwargs['dec'], unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity(50 / cluster_kwargs['dist'] * 180 / np.pi, u.deg)
        _ = Gaia.cone_search_async(coord, radius, output_file=os.path.join(save_dir, cluster + '.csv'),
                                   output_format='csv', dump_to_file=True, columns=cone_fields)
        print(f'done')

    Gaia.logout()


if __name__ == "__main__":
    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135']
    clusters_ = ['NGC_1605']

    data_path_ = '/home/matthijs/git/gaia_oc_amd/data'

    for cluster_ in clusters_:
        download_cone_file(cluster_, data_path_)
