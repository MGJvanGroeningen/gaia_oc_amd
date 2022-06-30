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


def flux_error_to_mag_error(row, band, zero_point):
    mean_mag_error = -2.5 * row[f'phot_{band}_mean_flux_error'] / (row[f'phot_{band}_mean_flux'] * np.log(10))
    mean_mag_error = np.sqrt(mean_mag_error**2 + zero_point**2)
    return mean_mag_error


def phot_g_mean_mag_error():
    sigma_g_0 = 0.0027553202

    def phot_g_mean_mag(row):
        return flux_error_to_mag_error(row, 'g', sigma_g_0)
    return phot_g_mean_mag


def bp_rp_error():
    sigma_bp_0 = 0.0027901700
    sigma_rp_0 = 0.0037793818

    def bp_rp_err(row):
        bp_mean_mag_error = flux_error_to_mag_error(row, 'bp', sigma_bp_0)
        rp_mean_mag_error = flux_error_to_mag_error(row, 'rp', sigma_rp_0)
        return np.sqrt(bp_mean_mag_error**2 + rp_mean_mag_error**2)
    return bp_rp_err


def radius(cluster):
    def calculate_radius(row):
        x, y = projected_coordinates(row, cluster)
        return np.sqrt(x ** 2 + y ** 2)
    return calculate_radius


def ellipse_isochrone_contacts(point, isochrone_pairs, x_scale, y_scale):
    x = point[0]
    y = point[1]

    x0s = isochrone_pairs[:, 0, 0] - x
    y0s = isochrone_pairs[:, 0, 1] - y
    x1s = isochrone_pairs[:, 1, 0] - x
    y1s = isochrone_pairs[:, 1, 1] - y

    delta_y = y1s - y0s
    delta_x = np.where(x1s - x0s == 0, 1, x1s - x0s)

    m = delta_y / delta_x
    c = y0s - m * x0s

    # determine the parameters of the source-centered ellipse that just hits the isochrone line section
    k = y_scale / x_scale

    # determine the coordinates of the contact point
    x_contacts = - m * c / (m ** 2 + k ** 2)
    y_contacts = m * x_contacts + c

    contact_points = np.concatenate(((x_contacts + x)[:, None], (y_contacts + y)[:, None]), axis=1)

    return contact_points


def source_isochrone_section_delta(point, deltas, isochrone_pairs):
    scale_factors = np.array([1 / deltas[0], 1 / deltas[1]])

    contact_points = ellipse_isochrone_contacts(point, isochrone_pairs, deltas[0], deltas[1])
    contact_point_delta = (contact_points - point) * scale_factors

    delta_1 = (isochrone_pairs[:, 0] - point) * scale_factors
    delta_2 = (isochrone_pairs[:, 1] - point) * scale_factors

    closest_isochrone_point_delta = np.where(np.tile(norm(delta_1, axis=1) < norm(delta_2, axis=1), (2, 1)).T,
                                             delta_1, delta_2)

    vectors = np.where(np.tile((np.min(isochrone_pairs[:, :, 0], axis=1) < contact_points[:, 0]) &
                               (np.max(isochrone_pairs[:, :, 0], axis=1) > contact_points[:, 0]), (2, 1)).T,
                       contact_point_delta,
                       closest_isochrone_point_delta)

    shortest_isochrone_vector = vectors[np.argmin(norm(vectors, axis=1))]
    return shortest_isochrone_vector


def isochrone_delta(cluster, isochrone, use_source_errors=False):
    bp_rp_max_d = cluster.bp_rp_delta
    gmag_max_d = cluster.g_delta
    source_errors = cluster.source_errors

    bp_rp_pairs = np.stack((isochrone['bp_rp'].values[:-1], isochrone['bp_rp'].values[1:]), axis=-1)
    gmag_pairs = np.stack((isochrone['phot_g_mean_mag'].values[:-1], isochrone['phot_g_mean_mag'].values[1:]), axis=-1)

    isochrone_pairs = np.stack((bp_rp_pairs, gmag_pairs), axis=-1)

    def isochrone_del(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']

        point = np.array([bp_rp, gmag])
        if use_source_errors:
            errors = np.array([source_errors * row['bp_rp_error'] + bp_rp_max_d,
                               source_errors * row['phot_g_mean_mag_error'] + gmag_max_d])
            delta = source_isochrone_section_delta(point, errors, isochrone_pairs)
        else:
            errors = np.array([bp_rp_max_d, gmag_max_d])
            delta = source_isochrone_section_delta(point, errors, isochrone_pairs)
        return delta

    return isochrone_del


def pm_distance(cluster, use_source_errors=False):
    cluster_pmra = cluster.pmra
    cluster_pmdec = cluster.pmdec

    cluster_pmra_e = cluster.pmra_delta
    cluster_pmdec_e = cluster.pmdec_delta
    source_errors = cluster.source_errors

    def pm_dist(row):
        pmra, pmdec = row['pmra'], row['pmdec']
        if use_source_errors:
            pmra_e = cluster_pmra_e + source_errors * row['pmra_error']
            pmdec_e = cluster_pmdec_e + source_errors * row['pmdec_error']
        else:
            pmra_e = cluster_pmra_e
            pmdec_e = cluster_pmdec_e
        pmra_d = pmra - cluster_pmra
        pmdec_d = pmdec - cluster_pmdec
        pm_d = np.sqrt((pmra_d / pmra_e)**2 + (pmdec_d / pmdec_e)**2)
        return pm_d

    return pm_dist


def plx_distance(cluster, use_source_errors=False):
    cluster_parallax = cluster.parallax

    cluster_plx_e_plus = cluster.plx_delta_plus
    cluster_plx_e_min = cluster.plx_delta_min
    source_errors = cluster.source_errors

    def plx_dist(row):
        plx = row['parallax']

        if use_source_errors:
            if plx < cluster.parallax:
                plx_e = cluster_plx_e_plus + source_errors * row['parallax_error']
            else:
                plx_e = cluster_plx_e_min + source_errors * row['parallax_error']
            plx_d = np.abs((plx - cluster_parallax) / plx_e)
        else:
            if plx < cluster.parallax:
                plx_e = cluster_plx_e_plus
            else:
                plx_e = cluster_plx_e_min
            plx_d = np.abs((plx - cluster_parallax) / plx_e)
        return plx_d

    return plx_dist


def add_features(subsets, feature_functions, feature_labels):
    for f, label in zip(feature_functions, feature_labels):
        for subset in subsets:
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            subset[label] = subset.apply(f, axis=1, result_type=result_type)


class Features:
    def __init__(self, train_features, cluster, isochrone):
        self.pm_d = pm_distance(cluster)
        self.plx_d = plx_distance(cluster)
        self.iso_del = isochrone_delta(cluster, isochrone)
        self.r = radius(cluster)
        self.phot_g_mean_mag_error = phot_g_mean_mag_error()
        self.bp_rp_error = bp_rp_error()

        self.train_feature_functions = [self.pm_d, self.plx_d, self.iso_del]
        self.aux_feature_functions = [self.r, self.phot_g_mean_mag_error, self.bp_rp_error]
        self.train_feature_labels = ['f_pm', 'f_plx', ['f_c', 'f_g']]
        self.aux_feature_labels = ['f_r', 'phot_g_mean_mag_error', 'bp_rp_error']

        self.feature_functions = self.aux_feature_functions + self.train_feature_functions
        self.feature_labels = self.aux_feature_labels + self.train_feature_labels

        self.train_features = train_features
