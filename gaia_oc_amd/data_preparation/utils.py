import numpy as np

from gaia_oc_amd.data_preparation.cluster import Cluster


def norm(arr):
    """Calculates the norm of the vector in the last dimension of an array.

    Args:
        arr (float, array): Array

    Returns:
        arr_norm (float, array): The vector norm or an array of vector norms

    """
    arr_norm = np.sqrt(np.sum(np.square(arr), axis=-1))
    return arr_norm


def extinction_magnitude_correction(a0, colour, danielski_parameters):
    """Calculates the magnitude correction due to interstellar extinction for a source with a given colour.

    Args:
        a0 (float, array): Extinction coefficient at the Gaia reference wavelength (550 nm)
        colour (float, array): BP-RP colour of the source
        danielski_parameters (float, tuple): Danielski parameters

    Returns:
        magnitude_correction (float): The correction in magnitude

    """
    c1, c2, c3, c4, c5, c6, c7 = danielski_parameters
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    magnitude_correction = k * a0
    return magnitude_correction


def extinction_correction(sources, cluster):
    """Corrects the magnitude and colour of a set of sources for interstellar extinction.

    Args:
        sources (Dataframe): A dataframe containing a set of sources
        cluster (Cluster): Cluster object

    """
    if 'a0' in sources.columns.to_list():
        a_v = sources['a0']
    else:
        a_v = cluster.a_v

    danielski_g_parameters = (0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099)
    danielski_bp_parameters = (1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043)
    danielski_rp_parameters = (0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006)

    extinction_g = extinction_magnitude_correction(a_v, sources['bp_rp'], danielski_g_parameters)
    extinction_bp = extinction_magnitude_correction(a_v, sources['bp_rp'], danielski_bp_parameters)
    extinction_rp = extinction_magnitude_correction(a_v, sources['bp_rp'], danielski_rp_parameters)

    sources['phot_g_mean_mag'] -= extinction_g
    sources['bp_rp'] -= extinction_bp - extinction_rp


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
    """Creates a function that can be applied to a dataframe of sources
    to calculate the G magnitude error.

    Returns:
        phot_g_mean_mag_error (function): Function that returns the G magnitude error

    """
    sigma_g_0 = 0.0027553202

    def phot_g_mean_mag_error(source):
        g_flux, g_flux_error = source[f'phot_g_mean_flux'], source[f'phot_g_mean_flux_error']
        return magnitude_error_from_flux_error(g_flux, g_flux_error, sigma_g_0)
    return phot_g_mean_mag_error


def bp_rp_error_function():
    """Creates a function that can be applied to a dataframe of sources
    to calculate the BP-RP colour error.

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


def projected_coordinates(ra, dec, cluster):
    """Calculates the projected coordinates x and y, relative to the cluster center, from
    the right ascension and declination.

    Args:
        ra (float, array): Right ascension
        dec (float, array): Declination
        cluster (Cluster): Cluster object (from which the cluster sky position and distance are retrieved)

    Returns:
        x (float, array): The projected x coordinate
        y (float, array): The projected y coordinate

    """
    rad_per_deg = np.pi / 180
    ra = ra * rad_per_deg
    dec = dec * rad_per_deg

    d = cluster.dist
    ra_c = cluster.ra * rad_per_deg
    dec_c = cluster.dec * rad_per_deg

    x = d * np.sin(ra - ra_c) * np.cos(dec)
    y = d * (np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c))
    return x, y


def line_segments_tangent_point(line_segments, point):
    """Finds the tangent points of a number of lines, defined by corresponding line segments,
    and the circles centered on a given point.

    Args:
        line_segments (float, array): An array of line segments with dimensions (n_lines, 2, 2),
            each line segment being defined by two 2-dimensional points.
        point (float, array): A 2-dimensional point

    Returns:
        tangent_coordinates (float, array): Coordinates of the tangent points

    """
    x = point[0]
    y = point[1]

    # Coordinates relative to point
    x0 = line_segments[:, 0, 0] - x
    y0 = line_segments[:, 0, 1] - y
    x1 = line_segments[:, 1, 0] - x
    y1 = line_segments[:, 1, 1] - y

    delta_y = y1 - y0
    delta_x = np.where(x1 - x0 == 0, 1e-6, x1 - x0)

    # Line parameters (slope and intercept)
    m = delta_y / delta_x
    c = y0 - m * x0

    # Relative tangent coordinates
    dx = - m * c / (m ** 2 + 1)
    dy = m * dx + c

    tangent_coordinates = np.stack((dx + x, dy + y), axis=-1)

    return tangent_coordinates


def shortest_vector_to_curve(point, line_segments, x_scale=1, y_scale=1):
    """Finds the vector that connects a point and a curve with the smallest norm. The curve
    is made up of a number of line segments.

    Args:
        point (float, array): A 2-dimensional point
        line_segments (float, array): An array of line segments with dimensions (n_segments, 2, 2),
            each line segment being defined by two 2-dimensional points.
        x_scale (float): Scale of the x dimension
        y_scale (float): Scale of the y dimension

    Returns:
        shortest_vec_to_curve (float, array): Components of the vector that connects a point
            and a curve with the smallest norm

    """
    n_line_segments = len(line_segments)

    # Optionally scale the space
    scale_factors = np.array([1 / x_scale, 1 / y_scale])
    scaled_point = point * scale_factors
    scaled_line_segments = line_segments * scale_factors

    # The closest point on a line segment to another (free) point, is either one of the two end points
    # of the line segment or the tangent point of the line segment and a circle surrounding the free point.
    line_segments_end_point_1 = scaled_line_segments[:, 0]
    line_segments_end_point_2 = scaled_line_segments[:, 0]
    tangent_points = line_segments_tangent_point(scaled_line_segments, scaled_point)

    # Define the vectors to all 3 possible points for each line segment
    vectors_to_line_segments = np.concatenate((np.expand_dims(line_segments_end_point_1 - scaled_point, axis=1),
                                               np.expand_dims(line_segments_end_point_2 - scaled_point, axis=1),
                                               np.expand_dims(tangent_points - scaled_point, axis=1)), axis=1)

    # Determine the index of the vector with the smallest norm for all line segments
    # (0 = end point 1, 1 = end point 2, 2 = tangent point)

    # First set the index for each line segment to either 0 or 1, depending on the closest end point.
    shortest_vector_to_line_segment_indices = np.argmin(norm(vectors_to_line_segments[:, :2]), axis=1)

    # If the tangent point is located between the end points, it is always the closest point
    tangent_point_is_closest = ((np.min(scaled_line_segments[:, :, 0], axis=1) < tangent_points[:, 0]) &
                                (tangent_points[:, 0] < np.max(scaled_line_segments[:, :, 0], axis=1)))
    shortest_vector_to_line_segment_indices[tangent_point_is_closest] = 2

    # Select the vector with the smallest norm for each line segment
    line_segment_indices = np.arange(n_line_segments)
    shortest_vector_to_line_segments = vectors_to_line_segments[line_segment_indices,
                                                                shortest_vector_to_line_segment_indices]

    # Select the vector that has the smallest norm among all line segments.
    shortest_vec_to_curve = shortest_vector_to_line_segments[np.argmin(norm(shortest_vector_to_line_segments))]
    return shortest_vec_to_curve


def add_columns(dataframes, functions, labels):
    """Adds a number of columns to a number dataframes.

    Args:
        dataframes (list, Dataframe): List of dataframes containing source data (e.g. members, non-members, etc.)
        functions (list, functions): List of functions which determine the column value.
        labels (list, str): List of labels for the added columns.
    """
    for dataframe in dataframes:
        for function, label in zip(functions, labels):
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            dataframe[label] = dataframe.apply(function, axis=1, result_type=result_type)
