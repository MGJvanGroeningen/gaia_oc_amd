import numpy as np
import pandas as pd

from dustapprox.literature import edr3
from dustapprox.models import PrecomputedModel


def isochrone_preprocessing(isochrone, dist, a0=0., colour='g_rp', oldest_stage=7, interpolation_density=5.,
                            oldest_stage_to_interpolate=5):
    """Calibrates the isochrone of a cluster for a given distance and extinction. Optionally exclude data beyond
    a given evolutionary stage and/or interpolate between isochrone data points. Returns only the colour and
    G magnitude of the isochrone.

    Args:
        isochrone (Dataframe): Dataframe containing the isochrone data points.
        dist (float): Distance (in parsec) to the cluster
        a0 (float): Extinction (in mag) at 550 nm
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        oldest_stage (int): Oldest evolutionary stage to include in the isochrone (default (7) is early ASG).
            See the FAQ at http://stev.oapd.inaf.it/cmd_3.7/faq.html
        interpolation_density (float): Fills the space between the isochrone data points such that the minimum distance
            is 1 / interpolation_density
        oldest_stage_to_interpolate (int): The oldest evolutionary stage for which to add linearly interpolated points
            to the isochrone.

    Returns:
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.

    """
    # Exclude data beyond the evolutionary stage of the early asymptotic giant branch
    if oldest_stage >= 0:
        processed_isochrone = isochrone[isochrone['label'] <= oldest_stage].copy().reset_index(drop=True)
    else:
        raise ValueError(f'The oldest stage must {oldest_stage} be a positive integer.')

    # Correct the isochrone magnitudes for extinction
    processed_isochrone = correct_for_extinction(processed_isochrone, a0)

    # Define colours
    processed_isochrone['bp_rp'] = processed_isochrone.loc[:, 'G_BPmag'] - processed_isochrone.loc[:, 'G_RPmag']
    processed_isochrone['g_rp'] = processed_isochrone.loc[:, 'phot_g_mean_mag'] - processed_isochrone.loc[:, 'G_RPmag']

    # Convert to apparent magnitude
    distance_modulus = 5 * np.log10(dist) - 5
    processed_isochrone['phot_g_mean_mag'] += distance_modulus

    # Fill sparse segments of the isochrone
    processed_isochrone = interpolate_isochrone(processed_isochrone, interpolation_density=interpolation_density,
                                                colour=colour, oldest_stage_to_interpolate=oldest_stage_to_interpolate)

    processed_isochrone = processed_isochrone[[colour, 'phot_g_mean_mag']]
    return processed_isochrone


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
    return k


def correct_for_extinction(isochrone, a0, mode='dustapprox_precomputed'):
    """Corrects the magnitude and colour of a set of sources for interstellar extinction.

    Args:
        isochrone (Dataframe): Dataframe containing
        a0 (float): Extinction at 550 nm
        mode (str): The method for calculating the extinction. Use 'danielski_2018', 'EDR3' or
            'dustapprox_precomputed'. Default is 'dustapprox_precomputed'.

    Returns:
        isochrone (Dataframe): Dataframe containing the isochrone data points with corrected magnitudes.
    """
    if a0 == 0:
        return isochrone

    kg, kbp, krp = np.zeros(len(isochrone)), np.zeros(len(isochrone)), np.zeros(len(isochrone))

    if mode == 'danielski_2018':
        danielski_g_parameters = (0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099)
        danielski_bp_parameters = (1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043)
        danielski_rp_parameters = (0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006)
        bp_rp = isochrone['bp_rp']
        kg = danielski_extinction(bp_rp, a0, danielski_g_parameters)
        kbp = danielski_extinction(bp_rp, a0, danielski_bp_parameters)
        krp = danielski_extinction(bp_rp, a0, danielski_rp_parameters)
    elif mode == 'EDR3':
        edr3ext = edr3.edr3_ext()
        ms_data = isochrone['label'] <= 1
        top_data = isochrone['label'] > 1
        for subset, flavor in zip([ms_data, top_data], ['ms', 'top']):
            bp_rp = isochrone[subset]['bp_rp']
            kg[subset] = edr3ext.from_bprp('kG', bp_rp, a0, flavor=flavor)
            kbp[subset] = edr3ext.from_bprp('kBP', bp_rp, a0, flavor=flavor)
            krp[subset] = edr3ext.from_bprp('kRP', bp_rp, a0, flavor=flavor)
    elif mode == 'dustapprox_precomputed':
        isochrone['teff'] = 10 ** isochrone['logTe']
        isochrone['A0'] = a0
        lib = PrecomputedModel()
        r = lib.find(passband='Gaia')
        model_g = lib.load_model(r[0], passband='GAIA_GAIA3.G')
        model_bp = lib.load_model(r[0], passband='GAIA_GAIA3.Gbp')
        model_rp = lib.load_model(r[0], passband='GAIA_GAIA3.Grp')
        kg = model_g.predict(isochrone)
        kbp = model_bp.predict(isochrone)
        krp = model_rp.predict(isochrone)

    isochrone['phot_g_mean_mag'] += kg * a0
    isochrone['G_BPmag'] += kbp * a0
    isochrone['G_RPmag'] += krp * a0
    return isochrone


def interpolate_isochrone(isochrone, interpolation_density=10., colour='g_rp', oldest_stage_to_interpolate=3):
    """Interpolates (linearly) between isochrone datapoints to fill sparse segments.

    Args:
        isochrone (Dataframe): Dataframe containing
        interpolation_density (float): Fills the space between the isochrone data points such that the minimum distance
            is 1 / isochrone_density
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        oldest_stage_to_interpolate (int): The oldest evolutionary stage for which to add linearly interpolated points
            to the isochrone.

    Returns:
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.
    """
    old_isochrone = isochrone[isochrone['label'] > oldest_stage_to_interpolate].copy()
    young_isochrone = isochrone[isochrone['label'] <= oldest_stage_to_interpolate].copy()
    if len(young_isochrone) > 0 and interpolation_density > 0:
        min_dist = 1 / interpolation_density
        n_young_isochrone = len(young_isochrone)
        young_isochrone_arr = young_isochrone[[colour, 'phot_g_mean_mag']].values
        deltas = np.diff(young_isochrone_arr, axis=0)
        dists = np.linalg.norm(deltas * np.array([8, 1]), axis=-1)
        sparse_segments = dists > min_dist
        if np.sum(sparse_segments) > 0:
            inserts = [np.stack((np.linspace(0, delta[0], max(int(dist // min_dist), 3)),
                                 np.linspace(0, delta[1], max(int(dist // min_dist), 3)))).T[1:-1]
                       for delta, dist in zip(deltas[sparse_segments], dists[sparse_segments])]
            sparse_segment_idx = np.arange(n_young_isochrone - 1)[sparse_segments] + 1
            sparse_segment_idx = np.concatenate([insert.shape[0] * [idx] for idx, insert in zip(sparse_segment_idx,
                                                                                                inserts)])
            inserts = np.concatenate(inserts) + young_isochrone_arr[sparse_segment_idx - 1]
            new_young_isochrone_arr = np.insert(young_isochrone_arr, sparse_segment_idx, inserts, axis=0)
            young_isochrone = pd.DataFrame(new_young_isochrone_arr, columns=[colour, 'phot_g_mean_mag'])

    isochrone = pd.concat((young_isochrone[[colour, 'phot_g_mean_mag']],
                           old_isochrone[[colour, 'phot_g_mean_mag']])).reset_index(drop=True)

    return isochrone
