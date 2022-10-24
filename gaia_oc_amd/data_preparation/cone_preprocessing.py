import numpy as np
from zero_point import zpt

from gaia_oc_amd.utils import add_columns


def danielski_extinction(bp_rp, a0, danielski_parameters):
    """Calculates the magnitude correction due to interstellar extinction for a source with a given colour.
    This extinction law is taken from Danielski et al. (2018)

    Args:
        bp_rp (float, array): BP-RP colour of the source
        a0 (float, array): Extinction coefficient at the Gaia reference wavelength (550 nm)
        danielski_parameters (float, tuple): Danielski parameters

    Returns:
        magnitude_correction (float): The correction in magnitude

    """
    c1, c2, c3, c4, c5, c6, c7 = danielski_parameters
    k = c1 + c2 * bp_rp + c3 * bp_rp ** 2 + c4 * bp_rp ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * bp_rp * a0
    return a0 * k


def gaia_extinctions(bp_rp, a0):
    """Corrects the magnitude and colour of a set of sources for interstellar extinction.

    Args:
        bp_rp (float, array): Array of colours
        a0 (float, array): Extinction at 550 nm, can either be a single value or an array

    """
    danielski_g_parameters = (0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099)
    danielski_bp_parameters = (1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043)
    danielski_rp_parameters = (0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006)

    a_g = danielski_extinction(bp_rp, a0, danielski_g_parameters)
    a_bp = danielski_extinction(bp_rp, a0, danielski_bp_parameters)
    a_rp = danielski_extinction(bp_rp, a0, danielski_rp_parameters)

    return a_g, a_bp, a_rp


def unsplit_sky_positions(sources, coordinate_system='icrs'):
    """Adjusts the sky positions of sources if the cluster overlaps the coordinate system zero point, which causes
    coordinate values of ~0 and ~360 and leads to sky position plots with split distributions.

    Args:
        sources (Dataframe): A dataframe containing a set of sources
        coordinate_system (str): The coordinate system

    """
    if coordinate_system == 'icrs':
        coordinate = 'ra'
        x = sources['ra']
    elif coordinate_system == 'galactic':
        coordinate = 'l'
        x = sources['l']
    else:
        coordinate = None
        x = None

    if x is not None:
        if x.max() - x.min() > 180:
            x = np.where(x > 180, x - 360, x)
            sources[coordinate] = x


def magnitude_error_from_flux_error(flux, flux_error, zero_point_error):
    """Calculates the error of the mean magnitude from the error of the mean flux.

    Args:
        flux (Series): Right ascension
        flux_error (Series): Right ascension
        zero_point_error (float): Error in the magnitude zero point

    Returns:
        mean_mag_error (float): The projected x coordinate

    """
    mean_mag_error = np.sqrt((-2.5 * flux_error / (flux * np.log(10)))**2 + zero_point_error**2)
    return mean_mag_error


def phot_g_mean_mag_error_function():
    """Creates a function that can be applied to a dataframe of sources to calculate the G magnitude error.

    Returns:
        phot_g_mean_mag_error (function): Function that returns the G magnitude error

    """
    sigma_g_0 = 0.0027553202

    def phot_g_mean_mag_error(source):
        g_flux, g_flux_error = source[f'phot_g_mean_flux'], source[f'phot_g_mean_flux_error']
        return magnitude_error_from_flux_error(g_flux, g_flux_error, sigma_g_0)
    return phot_g_mean_mag_error


def bp_rp_error_function():
    """Creates a function that can be applied to a dataframe of sources to calculate the BP-RP colour error.

    Returns:
        bp_rp_error (function): Function that returns the BP-RP colour error

    """
    sigma_bp_0 = 0.0027901700
    sigma_rp_0 = 0.0037793818

    def bp_rp_error(source):
        bp_flux, bp_flux_error = source[f'phot_bp_mean_flux'], source[f'phot_bp_mean_flux_error']
        rp_flux, rp_flux_error = source[f'phot_rp_mean_flux'], source[f'phot_rp_mean_flux_error']

        bp_mean_mag_error = magnitude_error_from_flux_error(bp_flux, bp_flux_error, sigma_bp_0)
        rp_mean_mag_error = magnitude_error_from_flux_error(rp_flux, rp_flux_error, sigma_rp_0)
        return np.sqrt(bp_mean_mag_error**2 + rp_mean_mag_error**2)
    return bp_rp_error


def cone_preprocessing(cone, a0=0., source_extinction_field=None):
    """Applies the following preprocessing steps on the cone search data.
     - Drop sources without colours
     - Correct the parallax of the cone sources for the parallax zero-point
     - Correct the magnitude and colour of cone sources for interstellar extinction
     - Adjust the coordinates to prevent a split in the sky position plot
     - Calculate magnitude and colour errors of the cone sources from their flux data
     - Set the default membership probability to zero

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.
        a0 (float, array): Extinction at 550 nm, can either be a single value or an array
        source_extinction_field (str): Optional extinction field if it is available per source in the cone data

    Returns:
        cone (Dataframe): Dataframe containing the (preprocessed) data of the sources in the cone search.
    """
    # Drop sources without colours
    bad_sources = np.arange(len(cone))[np.isnan(cone['bp_rp'].values)]
    cone = cone.drop(bad_sources).reset_index(drop=True)

    # Correct the parallax of the cone sources for the parallax zero-point
    zpt.load_tables()
    plx_zpts = zpt.get_zpt(cone['phot_g_mean_mag'].values, cone['nu_eff_used_in_astrometry'].values,
                           cone['pseudocolour'].values, cone['ecl_lat'].values,
                           cone['astrometric_params_solved'].values, _warnings=False)
    zpt_is_not_nan = ~np.isnan(plx_zpts)
    cone.loc[zpt_is_not_nan, 'parallax'] -= plx_zpts[zpt_is_not_nan]

    # Correct the magnitude and colour of cone sources for interstellar extinction
    if source_extinction_field is not None:
        a0 = cone[source_extinction_field]
    a_g, a_bp, a_rp = gaia_extinctions(cone['bp_rp'], a0)
    cone['phot_g_mean_mag'] -= a_g
    cone['bp_rp'] -= a_bp - a_rp

    # Adjust the coordinates to prevent a split in the sky position plot
    unsplit_sky_positions(cone, coordinate_system='galactic')
    unsplit_sky_positions(cone, coordinate_system='icrs')

    # Calculate magnitude and colour errors of the cone sources from their flux data
    add_columns([cone], [phot_g_mean_mag_error_function(), bp_rp_error_function()],
                ['phot_g_mean_mag_error', 'bp_rp_error'])

    # Set the default membership probability to zero
    cone['PMemb'] = 0
    return cone
