import numpy as np

from gaia_oc_amd.data_preparation.utils import norm


def projected_coordinates(members, cluster):
    rad_per_deg = np.pi / 180

    ra = members['ra'] * rad_per_deg
    dec = members['dec'] * rad_per_deg

    d = cluster.dist
    ra_c = cluster.ra * rad_per_deg
    dec_c = cluster.dec * rad_per_deg

    x = d * np.sin(ra - ra_c) * np.cos(dec)
    y = d * (np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c))
    return x, y


def flux_error_to_mag_error(row, band, zero_point_error):
    mean_mag_error = -2.5 * row[f'phot_{band}_mean_flux_error'] / (row[f'phot_{band}_mean_flux'] * np.log(10))
    mean_mag_error = np.sqrt(mean_mag_error**2 + zero_point_error**2)
    return mean_mag_error


def phot_g_mean_mag_error_function():
    sigma_g_0 = 0.0027553202

    def phot_g_mean_mag_error(row):
        return flux_error_to_mag_error(row, 'g', sigma_g_0)
    return phot_g_mean_mag_error


def bp_rp_error_function():
    sigma_bp_0 = 0.0027901700
    sigma_rp_0 = 0.0037793818

    def bp_rp_error(row):
        bp_mean_mag_error = flux_error_to_mag_error(row, 'bp', sigma_bp_0)
        rp_mean_mag_error = flux_error_to_mag_error(row, 'rp', sigma_rp_0)
        return np.sqrt(bp_mean_mag_error**2 + rp_mean_mag_error**2)
    return bp_rp_error


def radius_feature_function(cluster):
    def radius_feature(row):
        x, y = projected_coordinates(row, cluster)
        return np.sqrt(x ** 2 + y ** 2)
    return radius_feature


def line_segment_tangent_point(point, line_segments):
    x = point[0]
    y = point[1]

    x0 = line_segments[:, 0, 0] - x
    y0 = line_segments[:, 0, 1] - y
    x1 = line_segments[:, 1, 0] - x
    y1 = line_segments[:, 1, 1] - y

    delta_y = y1 - y0
    delta_x = np.where(x1 - x0 == 0, 1e-6, x1 - x0)

    m = delta_y / delta_x
    c = y0 - m * x0

    dx = - m * c / (m ** 2 + 1)
    dy = m * dx + c

    tangent_coordinates = np.stack((dx + x, dy + y), axis=-1)

    return tangent_coordinates


def shortest_vector_to_curve(point, line_segments, x_scale, y_scale):
    scale_factors = np.array([1 / x_scale, 1 / y_scale])

    point *= scale_factors
    scaled_line_segments = line_segments * scale_factors

    line_segments_end_point_1 = scaled_line_segments[:, 0]
    line_segments_end_point_2 = scaled_line_segments[:, 0]
    line_segments_tangent_point = line_segment_tangent_point(point, scaled_line_segments)

    vectors_to_line_segments = np.concatenate((np.expand_dims(line_segments_end_point_1 - point, axis=1),
                                               np.expand_dims(line_segments_end_point_2 - point, axis=1),
                                               np.expand_dims(line_segments_tangent_point - point, axis=1)), axis=1)

    shortest_vector_to_line_segments_index = np.argmin(norm(vectors_to_line_segments[:, :2], axis=2), axis=1)

    tangent_point_is_closest = ((np.min(scaled_line_segments[:, :, 0], axis=1) < line_segments_tangent_point[:, 0]) &
                                (line_segments_tangent_point[:, 0] < np.max(scaled_line_segments[:, :, 0], axis=1)))
    shortest_vector_to_line_segments_index[tangent_point_is_closest] = 2

    shortest_vector_to_line_segments = vectors_to_line_segments[np.arange(len(scaled_line_segments)),
                                                                shortest_vector_to_line_segments_index]
    shortest_vec_to_curve = shortest_vector_to_line_segments[np.argmin(norm(shortest_vector_to_line_segments, axis=1))]
    return shortest_vec_to_curve


def isochrone_features_function(cluster, isochrone, use_source_errors=False):
    bp_rp_max_d = cluster.bp_rp_delta
    gmag_max_d = cluster.g_delta
    source_errors = cluster.source_errors

    bp_rp_pairs = np.stack((isochrone['bp_rp'].values[:-1], isochrone['bp_rp'].values[1:]), axis=-1)
    gmag_pairs = np.stack((isochrone['phot_g_mean_mag'].values[:-1], isochrone['phot_g_mean_mag'].values[1:]), axis=-1)
    isochrone_sections = np.stack((bp_rp_pairs, gmag_pairs), axis=-1)

    def isochrone_features(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        bp_rp_e, gmag_e = bp_rp_max_d, gmag_max_d
        if use_source_errors:
            bp_rp_e += source_errors * row['bp_rp_error']
            gmag_e += source_errors * row['phot_g_mean_mag_error']
        source = np.array([bp_rp, gmag])
        shortest_vector_to_isochrone = shortest_vector_to_curve(source, isochrone_sections, bp_rp_e, gmag_e)
        return shortest_vector_to_isochrone

    return isochrone_features


def pm_feature_function(cluster, use_source_errors=False):
    cluster_pmra = cluster.pmra
    cluster_pmdec = cluster.pmdec

    cluster_pmra_e = cluster.pmra_delta
    cluster_pmdec_e = cluster.pmdec_delta
    source_errors = cluster.source_errors

    def pm_feature(row):
        pmra, pmdec = row['pmra'], row['pmdec']
        pmra_e, pmdec_e = cluster_pmra_e, cluster_pmdec_e
        if use_source_errors:
            pmra_e += source_errors * row['pmra_error']
            pmdec_e += source_errors * row['pmdec_error']
        pm_d = np.sqrt(((pmra - cluster_pmra) / pmra_e)**2 + ((pmdec - cluster_pmdec) / pmdec_e)**2)
        return pm_d

    return pm_feature


def plx_feature_function(cluster, use_source_errors=False):
    cluster_parallax = cluster.parallax

    cluster_plx_e_plus = cluster.plx_delta_plus
    cluster_plx_e_min = cluster.plx_delta_min
    source_errors = cluster.source_errors

    def plx_feature(row):
        plx = row['parallax']
        if plx < cluster.parallax:
            plx_e = cluster_plx_e_plus
        else:
            plx_e = cluster_plx_e_min
        if use_source_errors:
            plx_e += source_errors * row['parallax_error']
        plx_d = np.abs((plx - cluster_parallax) / plx_e)
        return plx_d

    return plx_feature


def add_features(subsets, feature_functions, feature_labels):
    for f, label in zip(feature_functions, feature_labels):
        for subset in subsets:
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            subset[label] = subset.apply(f, axis=1, result_type=result_type)


class Features:
    def __init__(self, train_features, cluster, isochrone):
        self.f_r = radius_feature_function(cluster)
        self.f_pm = pm_feature_function(cluster)
        self.f_plx = plx_feature_function(cluster)
        self.f_iso = isochrone_features_function(cluster, isochrone)
        self.phot_g_mean_mag_error = phot_g_mean_mag_error_function()
        self.bp_rp_error = bp_rp_error_function()

        self.train_feature_functions = [self.f_r, self.f_pm, self.f_plx, self.f_iso]
        self.aux_feature_functions = [self.phot_g_mean_mag_error, self.bp_rp_error]
        self.train_feature_labels = ['f_r', 'f_pm', 'f_plx', ['f_c', 'f_g']]
        self.aux_feature_labels = ['phot_g_mean_mag_error', 'bp_rp_error']

        self.feature_functions = self.aux_feature_functions + self.train_feature_functions
        self.feature_labels = self.aux_feature_labels + self.train_feature_labels

        self.train_features = train_features
