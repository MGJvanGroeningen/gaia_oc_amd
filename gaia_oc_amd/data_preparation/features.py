import numpy as np

from gaia_oc_amd.utils import projected_coordinates


def radius_feature_function(sources, cluster_ra, cluster_dec, cluster_dist):
    """Calculates the radius feature for a set of sources.

    Args:
        sources (Dataframe): Dataframe containing sources
        cluster_ra (float): Right ascension of the cluster
        cluster_dec (float): Declination of the cluster
        cluster_dist (float): Distance to the cluster

    Returns:
        radius_feature (function): Function that returns the radius feature of a source

    """
    ra, dec = sources['ra'], sources['dec']
    x, y = projected_coordinates(ra, dec, cluster_ra, cluster_dec, cluster_dist)
    f_r = np.sqrt(x ** 2 + y ** 2)
    return f_r


def pm_feature_function(sources, cluster_pmra, cluster_pmdec, delta_pm, use_source_errors=False,
                        source_error_weight=3.0, scale_features=False):
    """Calculates the proper motion feature for a set of sources.

    Args:
        sources (Dataframe): Dataframe containing sources
        cluster_pmra (float): Mean pmra of the cluster
        cluster_pmdec (float): Mean pmdec of the cluster
        delta_pm (float): Maximum proper motion separation
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.

    Returns:
        pm_feature (function): Function that returns the proper motion feature of a source

    """
    pmra_d = sources['pmra'] - cluster_pmra
    pmdec_d = sources['pmdec'] - cluster_pmdec
    if scale_features:
        pmra_scale, pmdec_scale = delta_pm, delta_pm
        if use_source_errors:
            pmra_scale += source_error_weight * sources['pmra_error']
            pmdec_scale += source_error_weight * sources['pmdec_error']
        pmra_d /= pmra_scale
        pmdec_d /= pmdec_scale
    f_pm = np.sqrt(pmra_d ** 2 + pmdec_d ** 2)
    return f_pm


def plx_feature_function(sources, cluster_parallax, delta_plx_plus, delta_plx_min, use_source_errors=False,
                         source_error_weight=3.0, scale_features=False):
    """Calculates the parallax feature for a set of sources.

    Args:
        sources (Dataframe): Dataframe containing sources
        cluster_parallax (float): Mean parallax of the cluster
        delta_plx_plus (float): Maximum parallax separation for sources closer to us than the cluster
        delta_plx_min (float): Maximum parallax separation for sources farther away from us than the cluster
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.

    Returns:
        plx_feature (function): Function that returns the parallax feature of a source

    """
    plx = sources['parallax']
    plx_d = plx - cluster_parallax
    if scale_features:
        plx_scale = np.where(plx < cluster_parallax, delta_plx_plus, delta_plx_min)
        if use_source_errors:
            plx_scale += source_error_weight * sources['parallax_error']
        plx_d /= plx_scale
    f_plx = plx_d
    return f_plx


def shortest_vector_to_curve(points, curve_points, scale_factors):
    """Calculates the shortest (scaled) vector from a (number of) N-dimensional point(s) to a curve defined
    by a sequence of points.

    Args:
        points (float, array): Array containing N-dimensional points. (n_points, N)
        curve_points (float, array): Array containing a sequence of N-dimensional points. (n_curve_points, N)
        scale_factors (float, array): Array containing scales corresponding to the points. (n_points, N)

    Returns:
        shortest_vec_to_curve (float, array): Array with the shortest vector to the curve for each point. (n_points, N)
    """
    n_points = points.shape[0]
    n_curve_points = curve_points.shape[0]

    points = np.tile(np.expand_dims(points, axis=1), (1, n_curve_points, 1))
    scale_factors = np.tile(np.expand_dims(scale_factors, axis=1), (1, n_curve_points, 1))
    vectors_to_curve = scale_factors * (curve_points - points)

    shortest_vec_ids = np.argmin(np.sqrt(np.einsum("kij,kij->ki", vectors_to_curve, vectors_to_curve)), axis=-1)
    shortest_vec_to_curve = vectors_to_curve[np.arange(n_points), shortest_vec_ids]
    return shortest_vec_to_curve


def isochrone_features_function(sources, isochrone, delta_c, delta_g, colour='bp_rp', use_source_errors=False,
                                source_error_weight=3.0, scale_features=False, sources_per_chunk=10000):
    """Calculates the isochrone features for a set of sources.

    Args:
        sources (Dataframe): Dataframe containing sources
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        delta_c (float): Maximum colour separation
        delta_g (float): Maximum magnitude separation
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.
        sources_per_chunk (int): Number of sources for which to calculate the shortest vector to the isochrone
            simultaneously.

    Returns:
        f_c (float, array): Array with the colour component of the shortest vector to the curve for each point
        f_g (float, array): Array with the magnitude component of the shortest vector to the curve for each point

    """
    n_sources = len(sources)
    isochrone_curve = np.stack((isochrone[colour].values, isochrone['phot_g_mean_mag'].values), axis=-1)
    c, g = np.array(sources[colour], ndmin=1), np.array(sources['phot_g_mean_mag'], ndmin=1)
    c_scale, g_scale = np.tile(delta_c, n_sources), np.tile(delta_g, n_sources)
    if use_source_errors:
        c_scale += source_error_weight * sources[colour + '_error']
        g_scale += source_error_weight * sources['phot_g_mean_mag_error']

    scale_factors = np.array([1 / c_scale, 1 / g_scale]).T
    # scaled_sources = np.array([c, g]).T * scale_factors.T
    scaled_sources = np.array([c, g]).T
    # scaled_isochrone_curve = isochrone_curve * scale_factors
    scaled_isochrone_curve = isochrone_curve

    n_chunks = n_sources // sources_per_chunk + 1
    shortest_vector_to_isochrone = np.zeros((n_sources, 2))
    for i in range(n_chunks):
        start = int(i * sources_per_chunk)
        stop = int((i + 1) * sources_per_chunk)
        shortest_vector_to_isochrone[start:stop] = shortest_vector_to_curve(scaled_sources[start:stop],
                                                                            scaled_isochrone_curve,
                                                                            scale_factors[start:stop])

    f_c, f_g = shortest_vector_to_isochrone[:, 0], shortest_vector_to_isochrone[:, 1]
    if not scale_features:
        f_c *= c_scale
        f_g *= g_scale
    return f_c, f_g


def add_features(sources, cluster, radius_feature=True, proper_motion_feature=True, parallax_feature=True,
                 isochrone_features=True):
    """Adds the values of the radius, proper motion, parallax and isochrone features to the given source dataframes.

    Args:
        sources (Dataframe, list): Dataframe or list of dataframes containing sources.
        cluster (Cluster): Cluster object
        radius_feature (bool): Whether to add the radius feature
        proper_motion_feature (bool): Whether to add the proper motion feature
        parallax_feature (bool): Whether to add the parallax feature
        isochrone_features (bool): Whether to add the isochrone features

    """
    if type(sources) != list:
        sources = [sources]

    for s in sources:
        if radius_feature:
            s['f_r'] = radius_feature_function(s, cluster.ra, cluster.dec, cluster.dist)
        if proper_motion_feature:
            s['f_pm'] = pm_feature_function(s, cluster.pmra, cluster.pmdec, cluster.delta_pm)
        if parallax_feature:
            s['f_plx'] = plx_feature_function(s, cluster.parallax, cluster.delta_plx_plus, cluster.delta_plx_min)
        if isochrone_features:
            s['f_c'], s['f_g'] = isochrone_features_function(s, cluster.isochrone, cluster.delta_c,
                                                             cluster.delta_g, colour=cluster.isochrone_colour)
