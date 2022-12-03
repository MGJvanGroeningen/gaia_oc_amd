import numpy as np

from gaia_oc_amd.data_preparation.features import pm_feature_function, plx_feature_function, \
    isochrone_features_function


def pm_candidate_condition(cluster_pmra, cluster_pmdec, pm_delta, source_error_weight=3.0):
    """Returns a function that can be applied on the cone dataframe to select candidates in proper motion space.

    Args:
        cluster_pmra (float): Mean pmra of the cluster
        cluster_pmdec (float): Mean pmdec of the cluster
        pm_delta (float): Maximum proper motion separation
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        can_condition (function): Candidate selection function
    """
    pm_feature = pm_feature_function(cluster_pmra, cluster_pmdec, pm_delta, use_source_errors=True,
                                     source_error_weight=source_error_weight, scale_features=True)

    def can_condition(row):
        candidate = False
        if pm_feature(row) < 1.0:
            candidate = True
        return candidate

    return can_condition


def plx_candidate_condition(cluster_parallax, plx_delta_plus, plx_delta_min, source_error_weight=3.0):
    """Returns a function that can be applied on the cone dataframe to select candidates in parallax space.

    Args:
        cluster_parallax (float): Mean parallax of the cluster
        plx_delta_plus (float): Maximum parallax separation for sources closer to us than the cluster
        plx_delta_min (float): Maximum parallax separation for sources farther away from us than the cluster
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        can_condition (function): Candidate selection function
    """
    plx_feature = plx_feature_function(cluster_parallax, plx_delta_plus, plx_delta_min, use_source_errors=True,
                                       source_error_weight=source_error_weight, scale_features=True)

    def can_condition(row):
        candidate = False
        if np.abs(plx_feature(row)) < 1.0:
            candidate = True
        return candidate

    return can_condition


def isochrone_candidate_condition(isochrone, c_delta=0.5, g_delta=1.5, colour='bp_rp', source_error_weight=3.0):
    """Returns a function that can be applied on the cone dataframe to select candidates based on their
    magnitude and colour.

    Args:
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        c_delta (float): Maximum colour separation
        g_delta (float): Maximum magnitude separation
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        can_condition (function): Candidate selection function
    """
    isochrone_features = isochrone_features_function(isochrone, c_delta, g_delta, colour=colour,
                                                     use_source_errors=True,
                                                     source_error_weight=source_error_weight,
                                                     scale_features=True)

    def can_condition(row):
        candidate = False
        if np.linalg.norm(isochrone_features(row)) < 1.0:
            candidate = True
        return candidate

    return can_condition


def candidate_conditions(cluster):
    """Returns a function that can be applied on the cone dataframe to select candidates based on their
    proper motion, parallax, magnitude and colour.

    Args:
        cluster (Cluster): Cluster object containing the cluster properties.

    Returns:
        can_condition (function): Candidate selection function
    """
    conditions = [plx_candidate_condition(cluster.parallax, cluster.delta_plx_plus, cluster.delta_plx_min,
                                          cluster.source_error_weight),
                  pm_candidate_condition(cluster.pmra, cluster.pmdec, cluster.delta_pm, cluster.source_error_weight),
                  isochrone_candidate_condition(cluster.isochrone, cluster.delta_c, cluster.delta_g,
                                                cluster.isochrone_colour,
                                                cluster.source_error_weight)]

    def can_condition(row):
        candidate = True
        for condition in conditions:
            satisfied_condition = condition(row)
            if not satisfied_condition:
                candidate = False
                break
        return candidate

    return can_condition
