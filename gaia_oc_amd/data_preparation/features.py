import numpy as np

from gaia_oc_amd.data_preparation.utils import projected_coordinates, shortest_vector_to_curve


def radius_feature_function(cluster):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their radius feature value.

    Args:
        cluster (Cluster): Cluster object

    Returns:
        radius_feature (function): Function that returns the radius feature of a source

    """
    def radius_feature(source):
        ra, dec = source['ra'], source['dec']
        x, y = projected_coordinates(ra, dec, cluster)
        return np.sqrt(x ** 2 + y ** 2)
    return radius_feature


def pm_feature_function(cluster, use_source_errors=False):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their proper motion feature value.

    Args:
        cluster (Cluster): Cluster object
        use_source_errors (bool): Whether to include the source errors in the feature. This is set to True
            when using the feature for candidate selection.

    Returns:
        pm_feature (function): Function that returns the proper motion feature of a source

    """
    cluster_pmra = cluster.pmra
    cluster_pmdec = cluster.pmdec

    cluster_pmra_e = cluster.pmra_delta
    cluster_pmdec_e = cluster.pmdec_delta

    if use_source_errors:
        source_error_weight = cluster.source_error_weight

    def pm_feature(source):
        pmra, pmdec = source['pmra'], source['pmdec']
        pmra_e, pmdec_e = cluster_pmra_e, cluster_pmdec_e
        if use_source_errors:
            pmra_e += source_error_weight * source['pmra_error']
            pmdec_e += source_error_weight * source['pmdec_error']
        pm_d = np.sqrt(((pmra - cluster_pmra) / pmra_e)**2 + ((pmdec - cluster_pmdec) / pmdec_e)**2)
        return pm_d

    return pm_feature


def plx_feature_function(cluster, use_source_errors=False):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their parallax feature value.

    Args:
        cluster (Cluster): Cluster object
        use_source_errors (bool): Whether to include the source errors in the feature. This is set to True
            when using the feature for candidate selection.

    Returns:
        plx_feature (function): Function that returns the parallax feature of a source

    """
    cluster_parallax = cluster.parallax

    cluster_plx_e_plus = cluster.plx_delta_plus
    cluster_plx_e_min = cluster.plx_delta_min

    if use_source_errors:
        source_error_weight = cluster.source_error_weight

    def plx_feature(source):
        plx = source['parallax']
        if plx < cluster.parallax:
            plx_e = cluster_plx_e_plus
        else:
            plx_e = cluster_plx_e_min
        if use_source_errors:
            plx_e += source_error_weight * source['parallax_error']
        plx_d = np.abs((plx - cluster_parallax) / plx_e)
        return plx_d

    return plx_feature


def isochrone_features_function(cluster, isochrone, use_source_errors=False):
    """Creates a function that can be applied to a dataframe of sources
    to calculate their isochrone feature values.

    Args:
        cluster (Cluster): Cluster object
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        use_source_errors (bool): Whether to include the source errors in the feature. This is set to True
            when using the feature for candidate selection.

    Returns:
        isochrone_features (function): Function that returns the isochrone features of a source

    """
    bp_rp_max_d = cluster.bp_rp_delta
    gmag_max_d = cluster.g_delta
    source_error_weight = cluster.source_error_weight

    bp_rp_pairs = np.stack((isochrone['bp_rp'].values[:-1], isochrone['bp_rp'].values[1:]), axis=-1)
    gmag_pairs = np.stack((isochrone['phot_g_mean_mag'].values[:-1], isochrone['phot_g_mean_mag'].values[1:]), axis=-1)
    isochrone_sections = np.stack((bp_rp_pairs, gmag_pairs), axis=-1)

    def isochrone_features(source):
        bp_rp, gmag = source['bp_rp'], source['phot_g_mean_mag']
        bp_rp_e, gmag_e = bp_rp_max_d, gmag_max_d
        if use_source_errors:
            bp_rp_e += source_error_weight * source['bp_rp_error']
            gmag_e += source_error_weight * source['phot_g_mean_mag_error']
        source = np.array([bp_rp, gmag])
        shortest_vector_to_isochrone = shortest_vector_to_curve(source, isochrone_sections, bp_rp_e, gmag_e)
        return shortest_vector_to_isochrone

    return isochrone_features


class Feature:
    """A small class for custom features.

    Args:
        function (function): A function that returns a feature value of a source
        label (str, list): A string or list of strings of the feature labels returned by the function.
    """
    def __init__(self, function, label):
        self.function = function
        self.label = label


class Features:
    """A class that stores custom feature functions in one place.

    Args:
        cluster (Cluster): Cluster object
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
    """
    def __init__(self, cluster, isochrone):
        self.f_r = Feature(radius_feature_function(cluster), 'f_r')
        self.f_pm = Feature(pm_feature_function(cluster), 'f_pm')
        self.f_plx = Feature(plx_feature_function(cluster), 'f_plx')
        self.f_iso = Feature(isochrone_features_function(cluster, isochrone), ['f_c', 'f_g'])

        self.features = [self.f_r, self.f_pm, self.f_plx, self.f_iso]
        self.update_after_sample_features = [self.f_pm, self.f_plx, self.f_iso]
        self.functions = [self.f_r.function, self.f_pm.function, self.f_plx.function, self.f_iso.function]
        self.labels = [self.f_r.label, self.f_pm.label, self.f_plx.label, self.f_iso.label]
