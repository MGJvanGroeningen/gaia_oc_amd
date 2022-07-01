import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.utils import commons


def query_catalog(catalog, columns, save_path, row_limit=-1, cluster_column=None, id_column=None,
                  prob_column=None):
    query = Vizier(catalog=catalog, columns=columns, row_limit=row_limit).query_constraints()[0]
    data = query.to_pandas()
    data = data.rename(columns={cluster_column: 'cluster', id_column: 'source_id', prob_column: 'PMemb'})
    data.to_csv(save_path)


def cone_search(clusters, data_dir, gaia_credentials_path, query_fields=None, table="gaiaedr3.gaia_source",
                cone_radius=60, pm_sigmas=10, plx_sigmas=10, verbose=False):
    Gaia.login(credentials_file=gaia_credentials_path)

    clusters_dir = os.path.join(data_dir, 'clusters')
    if not os.path.exists(clusters_dir):
        os.mkdir(clusters_dir)

    for cluster in clusters:
        output_dir = os.path.join(clusters_dir, cluster.name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, 'cone.vot.gz')

        coord = SkyCoord(ra=cluster.ra, dec=cluster.dec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity(cone_radius / cluster.dist * 180 / np.pi, u.deg)
        pmra, pmdec = cluster.pmra, cluster.pmdec
        pmra_e, pmdec_e = cluster.pmra_error, cluster.pmdec_error
        plx, plx_e = cluster.parallax, cluster.parallax_error

        if query_fields is not None:
            columns = ','.join(map(str, query_fields))
        else:
            query_fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec',
                            'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error',
                            'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
                            'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux',
                            'phot_g_mean_flux_error', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux_error',
                            'source_id',
                            'phot_g_mean_mag', 'bp_rp',
                            'l', 'b']
            columns = ','.join(map(str, query_fields))

        ra_hours, dec = commons.coord_to_radec(coord)
        ra = ra_hours * 15.0  # Converts to degrees
        radius_deg = commons.radius_to_unit(radius, unit='deg')

        query = """
                SELECT
                    {columns}
                FROM
                    {table_name}
                WHERE
                    1 = CONTAINS(POINT('ICRS', {ra_column}, {dec_column}), CIRCLE('ICRS', {ra}, {dec}, {radius}))
                    AND sqrt(power((pmra - {pmra}) / ({pmra_e}), 2) + power((pmdec - {pmdec}) / ({pmdec_e}), 2)) <= {pm_sigmas}
                    AND abs((parallax - {plx}) / ({plx_e})) <= {plx_sigmas}
                """.format(**{'columns': columns, 'table_name': table,
                              'ra_column': Gaia.MAIN_GAIA_TABLE_RA, 'dec_column': Gaia.MAIN_GAIA_TABLE_DEC,
                              'ra': ra, 'dec': dec, 'radius': radius_deg,
                              'pmra': pmra, 'pmdec': pmdec, 'pmra_e': pmra_e, 'pmdec_e': pmdec_e,
                              'pm_sigmas': pm_sigmas,
                              'plx': plx, 'plx_e': plx_e,
                              'plx_sigmas': plx_sigmas})

        print(f'Downloading {cluster.name} cone...', end=' ')
        _ = Gaia.launch_job_async(query=query,
                                  output_file=output_file,
                                  output_format="votable",
                                  verbose=verbose,
                                  dump_to_file=True)

    Gaia.logout()
