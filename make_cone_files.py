import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import os
import numpy as np
from data_filters import fields
from astroquery.utils import commons


def download_cone_file(cluster, data_paths):
    credentials_path = data_paths['credentials']
    Gaia.login(credentials_file=credentials_path)

    # Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2, default
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"  # Select early Data Release 3
    Gaia.ROW_LIMIT = -1

    cone_fields = fields['query'].copy()

    if not os.path.exists(data_paths['train_members']):
        raise ValueError(f'The cluster {cluster.name} does not exist!')

    print(f'Downloading {cluster.name} cone...', end=' ')
    coord = SkyCoord(ra=cluster.ra, dec=cluster.dec, unit=(u.degree, u.degree), frame='icrs')
    radius = u.Quantity(60 / cluster.dist * 180 / np.pi, u.deg)
    pmra, pmdec = cluster.pmra, cluster.pmdec
    pmra_e, pmdec_e = cluster.pmra_e, cluster.pmdec_e
    plx, plx_e = cluster.plx, cluster.plx_e

    output_file = data_paths['cone']
    columns = cone_fields

    ra_hours, dec = commons.coord_to_radec(coord)
    ra = ra_hours * 15.0  # Converts to degrees
    radius_deg = commons.radius_to_unit(radius, unit='deg')

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
            AND sqrt(power((pmra - {pmra}) / ({pmra_e}), 2) + power((pmdec - {pmdec}) / ({pmdec_e}), 2)) <= 10
            AND abs((parallax - {plx}) / ({plx_e})) <= 10
            """.format(**{'ra_column': Gaia.MAIN_GAIA_TABLE_RA,
                          'row_limit': "TOP {0}".format(Gaia.ROW_LIMIT) if Gaia.ROW_LIMIT > 0 else "",
                          'dec_column': Gaia.MAIN_GAIA_TABLE_DEC, 'columns': columns, 'ra': ra, 'dec': dec,
                          'radius': radius_deg, 'pmra': pmra, 'pmdec': pmdec, 'pmra_e': pmra_e, 'pmdec_e': pmdec_e,
                          'plx': plx, 'plx_e': plx_e,
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
    clusters_ = 'NGC_1605'

    data_path_ = '/home/matthijs/git/gaia_oc_amd/data'

    for cluster_ in clusters_:
        download_cone_file(cluster_, data_path_)
