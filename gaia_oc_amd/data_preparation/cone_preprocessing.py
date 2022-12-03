import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from zero_point import zpt


def init_spline(df, col_knots, col_coeff):
    ddff = df[[col_knots, col_coeff]].dropna()
    return interpolate.BSpline(ddff[col_knots], ddff[col_coeff], 3, extrapolate=False)


class Edr3MagUncertainty:
    def __init__(self, spline_csv):
        _df = pd.read_csv(spline_csv)
        splines = dict()
        splines['g'] = init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs = {'g': 200, 'bp': 20, 'rp': 20}

    def estimate(self, band, mag, nobs):
        mag = np.where(mag > 20.95, 20.95, mag)
        mag = np.where(mag < 4.05, 4.05, mag)
        return 10 ** (self.__splines[band](mag) - np.log10(np.sqrt(nobs) / np.sqrt(self.__nobs[band])))


def set_magnitude_errors(cone):
    u = Edr3MagUncertainty(os.path.join(os.path.dirname(__file__), 'LogErrVsMagSpline.csv'))
    cone['phot_g_mean_mag_error'] = u.estimate('g', cone['phot_g_mean_mag'], cone['phot_g_n_obs'])
    cone['phot_bp_mean_mag_error'] = u.estimate('bp', cone['phot_bp_mean_mag'], cone['phot_bp_n_obs'])
    cone['phot_rp_mean_mag_error'] = u.estimate('rp', cone['phot_rp_mean_mag'], cone['phot_rp_n_obs'])
    cone['bp_rp_error'] = np.sqrt(cone['phot_bp_mean_mag_error']**2 + cone['phot_rp_mean_mag_error']**2)
    cone['g_rp_error'] = np.sqrt(cone['phot_g_mean_mag_error']**2 + cone['phot_rp_mean_mag_error']**2)
    return cone


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


def cone_preprocessing(cone):
    """Applies the following preprocessing steps on the cone search data.
     - Drop sources without colours
     - Correct the parallax of the cone sources for the parallax zero-point
     - Correct the magnitude and colour of cone sources for interstellar extinction
     - Adjust the coordinates to prevent a split in the sky position plot
     - Calculate magnitude and colour errors of the cone sources from their flux data
     - Set the default membership probability to zero

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.

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

    # Set magnitude errors based on the expected error for a certain magnitude and number of observations.
    cone = set_magnitude_errors(cone)

    # Adjust the coordinates to prevent a split in the sky position plot
    unsplit_sky_positions(cone, coordinate_system='galactic')
    unsplit_sky_positions(cone, coordinate_system='icrs')

    # Set the default membership probability to zero
    cone['PMemb'] = 0
    return cone
