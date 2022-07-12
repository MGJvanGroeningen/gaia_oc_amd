from gaia_oc_amd.data_preparation.features import pm_feature_function, plx_feature_function, isochrone_features_function
from gaia_oc_amd.data_preparation.utils import norm


def pm_candidate_condition(cluster):
    """Returns a function that can be applied on the cone dataframe
    to select candidates in proper motion space.

    Args:
        cluster (Cluster): Cluster object containing the cluster properties.

    Returns:
        can_condition (function): Candidate selection function
    """
    pm_feature = pm_feature_function(cluster, use_source_errors=True)

    def can_condition(row):
        candidate = False
        if pm_feature(row) < 1.0:
            candidate = True
        return candidate

    return can_condition


def plx_candidate_condition(cluster):
    """Returns a function that can be applied on the cone dataframe
    to select candidates in parallax space.

    Args:
        cluster (Cluster): Cluster object containing the cluster properties.

    Returns:
        can_condition (function): Candidate selection function
    """
    plx_feature = plx_feature_function(cluster, use_source_errors=True)

    def can_condition(row):
        candidate = False
        if plx_feature(row) < 1.0:
            candidate = True
        return candidate

    return can_condition


def isochrone_candidate_condition(cluster, isochrone):
    """Returns a function that can be applied on the cone dataframe
    to select candidates based on their magnitude and colour.

    Args:
        cluster (Cluster): Cluster object containing the cluster properties.
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.

    Returns:
        can_condition (function): Candidate selection function
    """
    isochrone_features = isochrone_features_function(cluster, isochrone, use_source_errors=True)

    def can_condition(row):
        candidate = False
        if norm(isochrone_features(row)) < 1.0:
            candidate = True
        return candidate

    return can_condition


def candidate_conditions(cluster, isochrone):
    """Returns a function that can be applied on the cone dataframe
    to select candidates based on their proper motion, parallax, magnitude and colour.

    Args:
        cluster (Cluster): Cluster object containing the cluster properties.
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.

    Returns:
        can_condition (function): Candidate selection function
    """
    conditions = [plx_candidate_condition(cluster),
                  pm_candidate_condition(cluster),
                  isochrone_candidate_condition(cluster, isochrone)]

    def can_condition(row):
        candidate = True
        for condition in conditions:
            satisfied_condition = condition(row)
            if not satisfied_condition:
                candidate = False
                break
        return candidate

    return can_condition
