import numpy as np

from gaia_oc_amd.utils import projected_coordinates, add_columns


def _radius_feature_function(sources, cluster_ra, cluster_dec, cluster_dist):
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


def _pm_feature_function(sources, cluster_pmra, cluster_pmdec, delta_pm, use_source_errors=False,
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


def _plx_feature_function(sources, cluster_parallax, delta_plx_plus, delta_plx_min, use_source_errors=False,
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


def shortest_vector_to_curve(points, curve_points):
    """Calculates the shortest vector from a (number of) N-dimensional point(s) to a curve defined
    by a sequence of points.

    Args:
        points (float, array): Array containing N-dimensional points. (n_points, N)
        curve_points (float, array): Array containing a sequence of N-dimensional points. (n_curve_points, N)

    Returns:
        shortest_vec_to_curve (float, array): Array with the shortest vector to the curve for each point. (n_points, N)
    """
    n_points = points.shape[0]
    n_curve_points = curve_points.shape[0]

    points = np.tile(np.expand_dims(points, axis=1), (1, n_curve_points, 1))
    vectors_to_curve = curve_points - points
    shortest_vec_to_curve = vectors_to_curve[np.arange(n_points), np.argmin(np.sqrt(np.sum(np.square(vectors_to_curve),
                                                                                           axis=-1)), axis=-1)]
    return shortest_vec_to_curve


def _isochrone_features_function(sources, isochrone, delta_c, delta_g, colour='bp_rp', use_source_errors=False,
                                 source_error_weight=3.0, scale_features=False):
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

    Returns:
        f_c (float, array): Array with the colour component of the shortest vector to the curve for each point
        f_g (float, array): Array with the magnitude component of the shortest vector to the curve for each point

    """
    isochrone_curve = np.stack((isochrone[colour].values, isochrone['phot_g_mean_mag'].values), axis=-1)
    c, g = np.array(sources[colour], ndmin=1), np.array(sources['phot_g_mean_mag'], ndmin=1)
    c_scale, g_scale = delta_c, delta_g
    if use_source_errors:
        c_scale += source_error_weight * sources[colour + '_error']
        g_scale += source_error_weight * sources['phot_g_mean_mag_error']

    scale_factors = np.array([1 / c_scale, 1 / g_scale])
    scaled_sources = np.array([c, g]).T * scale_factors
    scaled_isochrone_curve = isochrone_curve * scale_factors

    shortest_vector_to_isochrone = shortest_vector_to_curve(scaled_sources, scaled_isochrone_curve)
    f_c, f_g = shortest_vector_to_isochrone[:, 0], shortest_vector_to_isochrone[:, 1]
    if not scale_features:
        f_c *= c_scale
        f_g *= g_scale
    return f_c, f_g


def radius_feature_function(cluster_ra, cluster_dec, cluster_dist):
    """Creates a function that can be applied to a dataframe of sources to calculate their radius feature value.

    Args:
        cluster_ra (float): Right ascension of the cluster
        cluster_dec (float): Declination of the cluster
        cluster_dist (float): Distance to the cluster

    Returns:
        radius_feature (function): Function that returns the radius feature of a source

    """
    def radius_feature(source):
        return _radius_feature_function(source, cluster_ra, cluster_dec, cluster_dist)
    return radius_feature


def pm_feature_function(cluster_pmra, cluster_pmdec, delta_pm, use_source_errors=False, source_error_weight=3.0,
                        scale_features=False):
    """Creates a function that can be applied to a dataframe of sources to calculate their proper motion feature value.

    Args:
        cluster_pmra (float): Mean pmra of the cluster
        cluster_pmdec (float): Mean pmdec of the cluster
        delta_pm (float): Maximum proper motion separation
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.

    Returns:
        pm_feature (function): Function that returns the proper motion feature of a source

    """
    def pm_feature(source):
        return _pm_feature_function(source, cluster_pmra, cluster_pmdec, delta_pm,
                                    use_source_errors=use_source_errors,
                                    source_error_weight=source_error_weight,
                                    scale_features=scale_features)

    return pm_feature


def plx_feature_function(cluster_parallax, delta_plx_plus, delta_plx_min, use_source_errors=False,
                         source_error_weight=3.0, scale_features=False):
    """Creates a function that can be applied to a dataframe of sources to calculate their parallax feature value.

    Args:
        cluster_parallax (float): Mean parallax of the cluster
        delta_plx_plus (float): Maximum parallax separation for sources closer to us than the cluster
        delta_plx_min (float): Maximum parallax separation for sources farther away from us than the cluster
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.

    Returns:
        plx_feature (function): Function that returns the parallax feature of a source

    """
    def plx_feature(source):
        return _plx_feature_function(source, cluster_parallax, delta_plx_plus, delta_plx_min,
                                     use_source_errors=use_source_errors,
                                     source_error_weight=source_error_weight,
                                     scale_features=scale_features)

    return plx_feature


def isochrone_features_function(isochrone, delta_c, delta_g, colour='bp_rp', use_source_errors=False,
                                source_error_weight=3.0, scale_features=False):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their isochrone feature values.

    Args:
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        delta_c (float): Maximum colour separation
        delta_g (float): Maximum magnitude separation
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.
        scale_features (bool): Whether to divide the features by their Deltas.

    Returns:
        isochrone_features (function): Function that returns the isochrone features of a source

    """
    def isochrone_features(source):
        f_c, f_g = _isochrone_features_function(source, isochrone, delta_c, delta_g,
                                                colour=colour,
                                                use_source_errors=use_source_errors,
                                                source_error_weight=source_error_weight,
                                                scale_features=scale_features)
        return np.stack((f_c[0], f_g[0])).T

    return isochrone_features


def add_features(sources, cluster, radius_feature=True, proper_motion_feature=True, parallax_feature=True,
                 isochrone_features=True, fast_mode=False):
    """Adds the values of the radius, proper motion, parallax and isochrone features to the given source dataframes.

    Args:
        sources (Dataframe, list): Dataframe or list of dataframes containing sources.
        cluster (Cluster): Cluster object
        radius_feature (bool): Whether to add the radius feature
        proper_motion_feature (bool): Whether to add the proper motion feature
        parallax_feature (bool): Whether to add the parallax feature
        isochrone_features (bool): Whether to add the isochrone features
        fast_mode (bool): If True, use a faster but more memory intensive method, which might crash for
            many (>~10^5) sources.

    """
    if type(sources) != list:
        sources = [sources]

    if fast_mode:
        for s in sources:
            if radius_feature:
                s['f_r'] = _radius_feature_function(s, cluster.ra, cluster.dec, cluster.dist)
            if proper_motion_feature:
                s['f_pm'] = _pm_feature_function(s, cluster.pmra, cluster.pmdec, cluster.delta_pm)
            if parallax_feature:
                s['f_plx'] = _plx_feature_function(s, cluster.parallax, cluster.delta_plx_plus, cluster.delta_plx_min)
            if isochrone_features:
                s['f_c'], s['f_g'] = _isochrone_features_function(s, cluster.isochrone, cluster.delta_c,
                                                                  cluster.delta_g, colour=cluster.isochrone_colour)
    else:
        feature_functions = []
        feature_labels = []
        if radius_feature:
            feature_functions.append(radius_feature_function(cluster.ra, cluster.dec, cluster.dist))
            feature_labels.append('f_r')
        if proper_motion_feature:
            feature_functions.append(pm_feature_function(cluster.pmra, cluster.pmdec, cluster.delta_pm))
            feature_labels.append('f_pm')
        if parallax_feature:
            feature_functions.append(plx_feature_function(cluster.parallax, cluster.delta_plx_plus,
                                                          cluster.delta_plx_min))
            feature_labels.append('f_plx')
        if isochrone_features:
            feature_functions.append(isochrone_features_function(cluster.isochrone, cluster.delta_c, cluster.delta_g,
                                                                 colour=cluster.isochrone_colour))
            feature_labels.append(['f_c', 'f_g'])

        add_columns(sources, feature_functions, feature_labels)
