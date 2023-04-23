import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from zero_point import zpt


def init_spline(df, col_knots, col_coeff):
    """Initializes the spline for the knots and coefficients of a given band.

    Args:
        df (DataFrame): Dataframe containing the spline parameters
        col_knots (str): Column name of the knots for a given band
        col_coeff (str): Column name of the coefficients for a given band

    Returns:
        spline (Bspline): Spline indicating the expected uncertainty for a given band.

    """
    ddff = df[[col_knots, col_coeff]].dropna()
    spline = interpolate.BSpline(ddff[col_knots], ddff[col_coeff], 3, extrapolate=False)
    return spline


class Edr3MagUncertainty:
    """Class for estimating the uncertainty of the magnitude in a given band. This is done by means of a precomputed
    spline, which indicates the expected uncertainty for a given mean magnitude after a fixed amount of
    observations.

    Reference: This research has made use of the tool provided by Gaia DPAC
    (https://www.cosmos.esa.int/web/gaia/dr3-software-tools) to reproduce (E)DR3 Gaia photometric uncertainties
    described in the GAIA-C5-TN-UB-JMC-031 technical note using data in Riello et al. (2021).

    Args:
        spline_csv (str): Path to the parameters of the precomputed splines.

    """
    def __init__(self, spline_csv):
        _df = pd.read_csv(spline_csv)
        splines = dict()
        splines['g'] = init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs = {'g': 200, 'bp': 20, 'rp': 20}

    def estimate(self, band, mag, nobs):
        """Estimates the uncertainty for a given magnitude in a band after a certain amount of observations. In the
        case that the magnitude exceeds the bounds of the spline, the source is attributed with the error as
        would be found for the boundary magnitude.

        Args:
            band (str): Name of the band ('g', 'bp', 'rp')
            mag (float, array): Array with the magnitudes of a number of sources
            nobs (float, int): The number of observations in the given band for each source

        Returns:
            mag_error (float, array): Array with the magnitude uncertainties for each source
        """
        mag = np.where(mag > 20.95, 20.95, mag)
        mag = np.where(mag < 4.05, 4.05, mag)
        mag_error = 10 ** (self.__splines[band](mag) - np.log10(np.sqrt(nobs) / np.sqrt(self.__nobs[band])))
        return mag_error


def set_magnitude_errors(cone):
    """This function estimates the uncertainty in the three magnitude bands and uses these to calculate the errors
    in the colours as well. The cone is returned with the calculated uncertainties as attributes.

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.

    Returns:
        cone (Dataframe): Dataframe the cone sources which have magnitude and colour errors as attributes.

    """
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


def cone_preprocessing(cone):
    """Applies the following preprocessing steps on the cone search data.
     - Drop sources with NaN values
     - Correct the parallax of the cone sources for the parallax zero-point
     - Calculate magnitude and colour errors of the cone sources from their flux data
     - Adjust the coordinates to prevent a split in the sky position plot
     - Set the default membership probability to zero

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.

    Returns:
        cone (Dataframe): Dataframe containing the (preprocessed) data of the sources in the cone search.
    """
    columns = cone.columns.to_list()
    cone = cone.dropna(subset=[col for col in columns if col not in ['nu_eff_used_in_astrometry',
                                                                     'pseudocolour']])
    cone = cone.query('~nu_eff_used_in_astrometry.isnull() | ~pseudocolour.isnull()')

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
