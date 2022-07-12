import os
from urllib.parse import urljoin

import numpy as np
import requests
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from lxml import etree
from tqdm import tqdm


def query_isochrone(save_dir, min_log_age=6.00, max_log_age=9.99, dlog_age=0.01, met=0.0152):
    """Downloads isochrone data for a range of isochrone ages and a fixed metallicity.

    Args:
        save_dir (str): Path to directory where the isochrone data will be stored.
        min_log_age (float): Minimum log(age) of the isochrone
        max_log_age (float): Maximum log(age) of the isochrone
        dlog_age (float): Step in log(age) of the isochrone
        met (float): Metallicity of the isochrones

    """
    website = 'http://stev.oapd.inaf.it/cgi-bin/cmd_3.6'

    form_kwargs = {'cmd_version': ['3.5'],
                   'track_parsec': 'parsec_CAF09_v1.2S',
                   'track_colibri': 'parsec_CAF09_v1.2S_S_LMC_08_web',
                   'track_postagb': 'no',
                   'n_inTPC': '10',
                   'eta_reimers': '0.2',
                   'kind_interp': ['1'],
                   'kind_postagb': ['-1'],
                   'photsys_file': 'YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat',
                   'photsys_version': 'YBCnewVega',
                   'dust_sourceM': 'dpmod60alox40',
                   'dust_sourceC': 'AMCSIC15',
                   'kind_mag': ['2'],
                   'kind_dust': ['0'],
                   'extinction_av': '0.0',
                   'extinction_coeff': 'constant',
                   'extinction_curve': 'cardelli',
                   'kind_LPV': '3',
                   'imf_file': 'tab_imf/imf_kroupa_orig.dat',
                   'isoc_isagelog': '1',
                   'isoc_lagelow': f'{min_log_age}',
                   'isoc_lageupp': f'{max_log_age}',
                   'isoc_dlage': f'{dlog_age}',
                   'isoc_ismetlog': '0',
                   'isoc_zlow': f'{met}',
                   'isoc_zupp': '0.03',
                   'isoc_dz': '0.0',
                   'output_kind': '0',
                   'output_evstage': ['1'],
                   'lf_maginf': '-15',
                   'lf_magsup': '20',
                   'lf_deltamag': '0.5',
                   'sim_mtot': '1.0e4',
                   'output_gzip': '0',
                   '.cgifields': ['track_colibri',
                                  'photsys_version',
                                  'output_gzip',
                                  'isoc_ismetlog',
                                  'isoc_isagelog',
                                  'track_postagb',
                                  'track_parsec',
                                  'extinction_coeff',
                                  'dust_sourceM',
                                  'kind_LPV',
                                  'extinction_curve',
                                  'output_kind',
                                  'dust_sourceC'],
                   'submit_form': 'Submit'}

    print('Querying isochrones...', end=' ')
    r = requests.post(website, data={**form_kwargs})
    p = etree.HTML(r.content)
    output_url = urljoin(website, p.xpath("//a[contains(text(), 'output')]/@href")[0])
    r.close()
    print('done')

    r = requests.get(output_url)
    block_size = 2 ** 16
    data_size = int(r.headers['content-length'])

    with open(os.path.join(save_dir, 'isochrones.dat'), 'wb') as f:
        for block in tqdm(r.iter_content(block_size), total=data_size // block_size + 1,
                          desc='Downloading isochrones...'):
            f.write(block)
    r.close()


def query_catalog(catalog, columns, save_path, cluster_column=None, id_column=None, prob_column=None, row_limit=-1):
    """Downloads data from a given Vizier catalogue. Can be used for the members and cluster parameters.

    Args:
        catalog (str): Identifier of the Vizier catalog
        columns (str, list): List of columns to be downloaded from the catalog
        save_path (str): Path where the catalog data is saved
        row_limit (int): Maximum number of rows to be downloaded from the catalog
        cluster_column (str): Name of the cluster name column in the (member) catalog data
        id_column (str): Name of the source identity column in the (member) catalog data
        prob_column (str): Name of the membership probability column in the (member) catalog data

    """
    query = Vizier(catalog=catalog, columns=columns, row_limit=row_limit).query_constraints()[0]
    data = query.to_pandas()
    data = data.rename(columns={cluster_column: 'cluster', id_column: 'source_id', prob_column: 'PMemb'})
    data.to_csv(save_path)


def cone_search(clusters, save_dir, gaia_credentials_path, table="gaiaedr3.gaia_source", query_columns=None,
                cone_radius=60., pm_sigmas=10., plx_sigmas=10., verbose=False):
    """Performs a cone search, centered on a specific cluster, on the data in the Gaia archive
    This results in a set of sources that includes the members, candidate members and informative non-members.

    Args:
        clusters (Cluster, list): A list of cluster objects
        save_dir (str): Path to the directory where the cone search data will be saved
        gaia_credentials_path (str): Path to a file that contains a username and password
            to login to the Gaia archive
        table (str): The data table from which to download the cone sources
        query_columns (str, list): Which columns to download
        cone_radius (float): The projected radius of the cone search in parsec
        pm_sigmas (float): How many standard deviations (of the cluster members) in proper motion
            cone sources may deviate from the mean
        plx_sigmas (float): How many standard deviations (of the cluster members) in parallax
            cone sources may deviate from the mean
        verbose (bool): Whether to print information about the download process.

    """
    Gaia.login(credentials_file=gaia_credentials_path)

    clusters_dir = os.path.join(save_dir, 'clusters')
    if not os.path.exists(clusters_dir):
        os.mkdir(clusters_dir)

    for cluster in clusters:
        output_dir = os.path.join(clusters_dir, cluster.name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, 'cone.vot.gz')

        ra, dec = cluster.ra, cluster.dec
        radius = cone_radius / cluster.dist * 180 / np.pi
        pmra, pmdec = cluster.pmra, cluster.pmdec
        pmra_e, pmdec_e = cluster.pmra_error, cluster.pmdec_error
        plx, plx_e = cluster.parallax, cluster.parallax_error
        plx_delta = abs(plx - 1000 / (1000 / plx - cone_radius))

        if query_columns is None:
            # Default columns
            columns = ['ra', 'dec', 'parallax', 'pmra', 'pmdec',
                       'parallax_error', 'pmra_error', 'pmdec_error',
                       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
                       'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux',
                       'phot_g_mean_flux_error', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux_error',
                       'source_id',
                       'phot_g_mean_mag', 'bp_rp',
                       'l', 'b']
        else:
            columns = query_columns
        columns = ','.join(map(str, columns))

        query = """
                SELECT
                    {columns}
                FROM
                    {table_name}
                WHERE
                    1 = CONTAINS(POINT('ICRS', {ra_column}, {dec_column}), CIRCLE('ICRS', {ra}, {dec}, {radius}))
                    AND sqrt(power((pmra - {pmra}) / ({pmra_e}), 2) + power((pmdec - {pmdec}) / ({pmdec_e}), 2)) <= {pm_sigmas}
                    AND abs((parallax - {plx}) / ({plx_e} + {plx_delta} / {plx_sigmas})) <= {plx_sigmas}
                """.format(**{'columns': columns, 'table_name': table,
                              'ra_column': Gaia.MAIN_GAIA_TABLE_RA, 'dec_column': Gaia.MAIN_GAIA_TABLE_DEC,
                              'ra': ra, 'dec': dec, 'radius': radius,
                              'pmra': pmra, 'pmdec': pmdec, 'pmra_e': pmra_e, 'pmdec_e': pmdec_e,
                              'pm_sigmas': pm_sigmas,
                              'plx': plx, 'plx_e': plx_e, 'plx_delta': plx_delta,
                              'plx_sigmas': plx_sigmas})

        print(f'Downloading {cluster.name} cone...', end=' ')
        _ = Gaia.launch_job_async(query=query,
                                  output_file=output_file,
                                  output_format="votable",
                                  verbose=verbose,
                                  dump_to_file=True)

    Gaia.logout()
