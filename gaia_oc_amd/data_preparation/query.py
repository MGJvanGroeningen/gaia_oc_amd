import os
import numpy as np
import requests
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from lxml import etree
from tqdm import tqdm
from urllib.parse import urljoin


def query_isochrone(save_path, log_age_min=6.00, log_age_max=9.99, log_age_step=0.01, metal_frac=0.0152):
    """Downloads isochrone data for a range of isochrone ages and a fixed metallicity/metal fraction.

    Args:
        save_path (str): Path to where the isochrone data will be stored.
        log_age_min (float): Minimum log(age) of the isochrone
        log_age_max (float): Maximum log(age) of the isochrone
        log_age_step (float): Step in log(age) of the isochrone
        metal_frac (float): Metal fraction (Z) of the isochrones

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
                   'isoc_lagelow': f'{log_age_min}',
                   'isoc_lageupp': f'{log_age_max}',
                   'isoc_dlage': f'{log_age_step}',
                   'isoc_ismetlog': '0',
                   'isoc_zlow': f'{metal_frac}',
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

    with open(save_path, 'wb') as f:
        for block in tqdm(r.iter_content(block_size), total=data_size // block_size + 1,
                          desc='Downloading isochrones...'):
            f.write(block)
    r.close()




def query_vizier_catalog(catalog, save_path='.', columns=None, new_column_names=None, row_limit=-1):
    """Downloads data from a given Vizier catalogue. Can be used for the members and cluster parameters.

    Args:
        catalog (str): Identifier of the Vizier catalog
        save_path (str): Path where the catalog data is saved
        columns (str, list): List of columns to be downloaded from the catalog
        new_column_names (dict): Dictionary for renaming the dataframe columns in the format
            {'old_name': 'new_name', etc...}
        row_limit (int): Maximum number of rows to be downloaded from the catalog

    """
    if columns is None:
        columns = ["*"]
    query = Vizier(catalog=catalog, columns=columns, row_limit=row_limit).query_constraints()[0]
    data = query.to_pandas().rename(columns=new_column_names)
    data.to_csv(save_path)


def cone_search(cluster, save_dir, gaia_credentials_path, table="gaiadr3.gaia_source", query_columns=None,
                cone_radius=50., pm_sigmas=10., plx_sigmas=10.):
    """Performs a cone search, centered on a specific cluster, on the data in the Gaia archive
    This results in a set of sources that includes the members, candidate members and informative non-members.

    Args:
        cluster (Cluster): A Cluster object
        save_dir (str): Path to the directory where the cone search data will be saved
        gaia_credentials_path (str): Path to a file that contains a username and password
            to log in to the Gaia archive
        table (str): The data table from which to download the cone sources
        query_columns (str, list): Which columns to download
        cone_radius (float): The projected radius of the cone search in parsec
        pm_sigmas (float): How many standard deviations (of the cluster members) in proper motion
            cone sources may deviate from the mean
        plx_sigmas (float): How many standard deviations (of the cluster members) in parallax
            cone sources may deviate from the mean
    """
    Gaia.login(credentials_file=gaia_credentials_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cluster_dir = os.path.join(save_dir, cluster.name)
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)
    cone_file = os.path.join(cluster_dir, 'cone.vot.gz')

    ra, dec = cluster.ra, cluster.dec
    if ra < 0:
        ra += 360
    radius = cone_radius / cluster.dist * 180 / np.pi
    pmra, pmdec = cluster.pmra, cluster.pmdec
    pmra_e, pmdec_e = cluster.pmra_error, cluster.pmdec_error
    pm_e = np.sqrt((pmra_e**2 + pmdec_e**2) / 2)
    plx, plx_e = cluster.parallax, cluster.parallax_error

    if query_columns is None:
        # Default columns
        columns = ['ra', 'dec', 'parallax', 'pmra', 'pmdec',
                   'parallax_error', 'pmra_error', 'pmdec_error',
                   'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr',
                   'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux',
                   'phot_g_mean_flux_error', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux_error',
                   'source_id',
                   'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                   'bp_rp', 'g_rp',
                   'l', 'b',
                   'nu_eff_used_in_astrometry', 'pseudocolour', 'ecl_lat', 'astrometric_params_solved',
                   'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    else:
        columns = query_columns
    columns = ','.join(map(str, columns))

    plx_bound = plx_sigmas * plx_e
    pm_bound = pm_sigmas * pm_e

    query = """
            SELECT
                {columns}
            FROM
                {table_name}
            WHERE
                1 = CONTAINS(POINT('ICRS', {ra_column}, {dec_column}), CIRCLE('ICRS', {ra}, {dec}, {radius}))
                AND abs(parallax - {plx}) <= {plx_bound}
                AND sqrt(power(pmra - {pmra}, 2) + power(pmdec - {pmdec}, 2)) <= {pm_bound}
            """.format(**{'columns': columns, 'table_name': table,
                          'ra_column': Gaia.MAIN_GAIA_TABLE_RA, 'dec_column': Gaia.MAIN_GAIA_TABLE_DEC,
                          'ra': ra, 'dec': dec, 'radius': radius,
                          'pmra': pmra, 'pmdec': pmdec, 'pm_bound': pm_bound,
                          'plx': plx, 'plx_bound': plx_bound})

    print(f'Downloading {cluster.name} cone...', end=' ')
    _ = Gaia.launch_job_async(query=query,
                              output_file=cone_file,
                              output_format="votable",
                              verbose=False,
                              dump_to_file=True)

    Gaia.logout()
