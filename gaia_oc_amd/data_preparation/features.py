import numpy as np

from gaia_oc_amd.utils import projected_coordinates, add_columns


def radius_feature_function(cluster_ra, cluster_dec, cluster_dist):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their radius feature value.

    Args:
        cluster_ra (float): Right ascension of the cluster
        cluster_dec (float): Declination of the cluster
        cluster_dist (float): Distance to the cluster

    Returns:
        radius_feature (function): Function that returns the radius feature of a source

    """
    def radius_feature(source):
        ra, dec = source['ra'], source['dec']
        x, y = projected_coordinates(ra, dec, cluster_ra, cluster_dec, cluster_dist)
        return np.sqrt(x ** 2 + y ** 2)
    return radius_feature


def pm_feature_function(cluster_pmra, cluster_pmdec, delta_pm, use_source_errors=False, source_error_weight=3.0):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their proper motion feature value.

    Args:
        cluster_pmra (float): Mean pmra of the cluster
        cluster_pmdec (float): Mean pmdec of the cluster
        delta_pm (float): Maximum proper motion separation
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        pm_feature (function): Function that returns the proper motion feature of a source

    """
    def pm_feature(source):
        pmra, pmdec = source['pmra'], source['pmdec']
        pmra_scale, pmdec_scale = delta_pm, delta_pm
        if use_source_errors:
            pmra_scale += source_error_weight * source['pmra_error']
            pmdec_scale += source_error_weight * source['pmdec_error']
        pm_d = np.sqrt(((pmra - cluster_pmra) / pmra_scale)**2 + ((pmdec - cluster_pmdec) / pmdec_scale)**2)
        return pm_d

    return pm_feature


def plx_feature_function(cluster_parallax, delta_plx_plus, delta_plx_min, use_source_errors=False,
                         source_error_weight=3.0):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their parallax feature value.

    Args:
        cluster_parallax (float): Mean parallax of the cluster
        delta_plx_plus (float): Maximum parallax separation for sources closer to us than the cluster
        delta_plx_min (float): Maximum parallax separation for sources farther away from us than the cluster
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        plx_feature (function): Function that returns the parallax feature of a source

    """
    def plx_feature(source):
        plx = source['parallax']
        if plx < cluster_parallax:
            plx_scale = delta_plx_plus
        else:
            plx_scale = delta_plx_min
        if use_source_errors:
            plx_scale += source_error_weight * source['parallax_error']
        plx_d = np.abs((plx - cluster_parallax) / plx_scale)
        return plx_d

    return plx_feature


def shortest_vector_to_line(point, a, x0, eps=1e-8):
    """Calculates the shortest vector from a free point to one or more straight lines defined by two vectors, a
    and x0. For a line with parameterization: x = a * t + x0, with a and x0 both N-dimensional,
    we calculate the t for which d(norm(point - x)) / dt = 0 and use it to find the point on the line
    which is closest to the free point.

    Args:
        point (float, array): An N-dimensional point
        a (float, array): Vector(s) parallel the line(s) with dimensions (N) or (n_lines, N)
        x0 (float, array): Point(s) on the line(s) with dimensions (N) or (n_lines, N)
        eps (float): Epsilon to prevent division by zero

    Returns:
        shortest_vec_to_line (float, array): Shortest vector(s) from the point x to the line(s)

    """
    closest_point_t = np.sum(a * (point - x0), axis=-1) / np.maximum(np.linalg.norm(a, axis=-1), eps)
    closest_point = a * np.tile(closest_point_t[..., None], point.shape[0]) + x0
    shortest_vec_to_line = closest_point - point
    return shortest_vec_to_line


def shortest_vector_to_curve(point, curve):
    """Finds the vector that connects a point and a curve with the smallest norm. The curve is defined by
    a sequence points.

    Args:
        point (float, array): A N-dimensional point
        curve (float, array): An array of N-dimensional points with dimensions (n_points, N)

    Returns:
        shortest_vec_to_curve (float, array): Components of the vector that connects a point
            and a curve with the smallest norm

    """
    n_line_segments = curve.shape[0] - 1

    # The closest point on a line segment to another (free) point, is either one of the two end points
    # or lies somewhere in between, in which case the point is also the closest point on the line defined
    # by the line segment
    line_segments_point_1 = curve[:-1]
    line_segments_point_2 = curve[1:]
    shortest_vec_to_line = shortest_vector_to_line(point, line_segments_point_2 - line_segments_point_1,
                                                   line_segments_point_1)

    # Define the vectors to all 3 possible closest points for each line segment
    vectors_to_line_segments = np.concatenate((np.expand_dims(line_segments_point_1 - point, axis=1),
                                               np.expand_dims(line_segments_point_2 - point, axis=1),
                                               np.expand_dims(shortest_vec_to_line, axis=1)), axis=1)

    # Determine the index of the vector with the smallest norm for all line segments
    # (0 = end point 1, 1 = end point 2, 2 = tangent point)
    shortest_vec_indices = np.zeros(n_line_segments, dtype=int)

    # If the closest point on the line is located between the end points of the line segment,
    # it is always the closest point
    line_segments_x = np.stack((line_segments_point_1[:, 0], line_segments_point_2[:, 0]), axis=-1)
    x_min = np.min(line_segments_x, axis=1)
    x_max = np.max(line_segments_x, axis=1)
    x_t = point[0] + shortest_vec_to_line[:, 0]
    tangent_point_is_closest = (x_min < x_t) & (x_t < x_max)
    shortest_vec_indices[tangent_point_is_closest] = 2

    # Set the index for the remaining line segments to either 0 or 1, depending on the closest end point.
    shortest_vec_indices[~tangent_point_is_closest] = np.argmin(np.linalg.norm(
        vectors_to_line_segments[~tangent_point_is_closest, :2], axis=-1), axis=1)

    # Select the vector with the smallest norm for each line segment
    shortest_vector_to_line_segments = vectors_to_line_segments[np.arange(n_line_segments), shortest_vec_indices]

    # Select the vector that has the smallest norm among all line segments.
    shortest_vec_index = np.argmin(np.linalg.norm(shortest_vector_to_line_segments, axis=-1))
    shortest_vec_to_curve = shortest_vector_to_line_segments[shortest_vec_index]
    return shortest_vec_to_curve


def isochrone_features_function(isochrone, delta_c=0.5, delta_g=1.5, use_source_errors=False, source_error_weight=3.0):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their isochrone feature values.

    Args:
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        delta_c (float): Maximum colour separation
        delta_g (float): Maximum magnitude separation
        use_source_errors (bool): Whether to include the source errors in the feature.
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        isochrone_features (function): Function that returns the isochrone features of a source

    """
    isochrone_curve = np.stack((isochrone['bp_rp'].values, isochrone['phot_g_mean_mag'].values), axis=-1)

    def isochrone_features(source):
        c, g = source['bp_rp'], source['phot_g_mean_mag']
        c_scale, g_scale = delta_c, delta_g
        if use_source_errors:
            c_scale += source_error_weight * source['bp_rp_error']
            g_scale += source_error_weight * source['phot_g_mean_mag_error']

        scale_factors = np.array([1 / c_scale, 1 / g_scale])
        scaled_source = np.array([c, g]) * scale_factors
        scaled_isochrone_curve = isochrone_curve * scale_factors
        shortest_vector_to_isochrone = shortest_vector_to_curve(scaled_source, scaled_isochrone_curve)
        return shortest_vector_to_isochrone

    return isochrone_features


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

    feature_functions = []
    feature_labels = []
    if radius_feature:
        feature_functions.append(radius_feature_function(cluster.ra, cluster.dec, cluster.dist))
        feature_labels.append('f_r')
    if proper_motion_feature:
        feature_functions.append(pm_feature_function(cluster.pmra, cluster.pmdec, cluster.delta_pm))
        feature_labels.append('f_pm')
    if parallax_feature:
        feature_functions.append(plx_feature_function(cluster.parallax, cluster.delta_plx_plus, cluster.delta_plx_min))
        feature_labels.append('f_plx')
    if isochrone_features:
        feature_functions.append(isochrone_features_function(cluster.isochrone, cluster.delta_c, cluster.delta_g))
        feature_labels.append(['f_c', 'f_g'])

    add_columns(sources, feature_functions, feature_labels)
