import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import os
import numpy as np
from data_filters import get_cluster_parameters, fields
from astroquery.utils import commons


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

    cone_fields = fields['all'].copy()
    cone_fields.remove('PMemb')

    for cluster in clusters:
        if not os.path.exists(os.path.join(data_path, 'members', cluster + '.csv')):
            raise ValueError(f'The cluster {cluster} does not exist!')

        print(f'Downloading {cluster} cone...', end=' ')
        cluster_kwargs = get_cluster_parameters(param_path, cluster)
        coord = SkyCoord(ra=cluster_kwargs['ra'], dec=cluster_kwargs['dec'], unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity(60 / cluster_kwargs['dist'] * 180 / np.pi, u.deg)
        pmra, pmdec = cluster_kwargs['pmra'], cluster_kwargs['pmdec']

        output_file = os.path.join(save_dir, cluster + '.vot.gz')
        columns = cone_fields

        raHours, dec = commons.coord_to_radec(coord)
        ra = raHours * 15.0  # Converts to degrees
        radiusDeg = commons.radius_to_unit(radius, unit='deg')

        if columns:
            columns = ','.join(map(str, columns))
        else:
            columns = "*"

        query = """
                SELECT
                      {row_limit}
                      {columns},
                      DISTANCE(
                        POINT('ICRS', {ra_column}, {dec_column}),
                        POINT('ICRS', {ra}, {dec})
                      ) AS dist
                FROM
                  {table_name}
                WHERE
                  1 = CONTAINS(
                    POINT('ICRS', {ra_column}, {dec_column}),
                    CIRCLE('ICRS', {ra}, {dec}, {radius})
                  )
                AND sqrt(power(pmra - {pmra}, 2) + power(pmdec - {pmdec}, 2)) < 2.0
                """.format(**{'ra_column': Gaia.MAIN_GAIA_TABLE_RA,
                              'row_limit': "TOP {0}".format(Gaia.ROW_LIMIT) if Gaia.ROW_LIMIT > 0 else "",
                              'dec_column': Gaia.MAIN_GAIA_TABLE_DEC, 'columns': columns, 'ra': ra, 'dec': dec,
                              'radius': radiusDeg, 'pmra': pmra, 'pmdec': pmdec,
                              'table_name': Gaia.MAIN_GAIA_TABLE})

        _ = Gaia.launch_job_async(query=query,
                                  output_file=output_file,
                                  output_format="votable",
                                  verbose=False,
                                  dump_to_file=True,
                                  background=False)
        print(f'done')

    Gaia.logout()


if __name__ == "__main__":
    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135']
    clusters_ = ['NGC_1605']

    data_path_ = '/home/matthijs/git/gaia_oc_amd/data'

    for cluster_ in clusters_:
        download_cone_file(cluster_, data_path_)
