import numpy as np

from gaia_oc_amd.data_preparation.features import pm_feature_function, plx_feature_function, isochrone_features_function


def pm_candidate_condition(sources, cluster_pmra, cluster_pmdec, pm_delta, source_error_weight=3.0):
    """Returns a boolean array indicating which of the sources are labeled as candidates based on their proper motion.

    Args:
        sources (Dataframe): Dataframe containing sources
        cluster_pmra (float): Mean pmra of the cluster
        cluster_pmdec (float): Mean pmdec of the cluster
        pm_delta (float): Maximum proper motion separation
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        passed_condition_indices (bool, array): Boolean array indicating which sources satisfied the condition.
    """
    f_pm = pm_feature_function(sources, cluster_pmra, cluster_pmdec, pm_delta, use_source_errors=True,
                               source_error_weight=source_error_weight, scale_features=True)
    passed_condition_indices = f_pm.to_numpy() < 1.0
    return passed_condition_indices


def plx_candidate_condition(sources, cluster_parallax, plx_delta_plus, plx_delta_min, source_error_weight=3.0):
    """Returns a boolean array indicating which of the sources are labeled as candidates based on their parallax.

    Args:
        sources (Dataframe): Dataframe containing sources
        cluster_parallax (float): Mean parallax of the cluster
        plx_delta_plus (float): Maximum parallax separation for sources closer to us than the cluster
        plx_delta_min (float): Maximum parallax separation for sources farther away from us than the cluster
        source_error_weight (float): How many source errors are added to the maximum separations.

    Returns:
        passed_condition_indices (bool, array): Boolean array indicating which sources satisfied the condition.
    """
    f_plx = plx_feature_function(sources, cluster_parallax, plx_delta_plus,
                                 plx_delta_min, use_source_errors=True,
                                 source_error_weight=source_error_weight, scale_features=True)
    passed_condition_indices = np.abs(f_plx.to_numpy()) < 1.0
    return passed_condition_indices


def isochrone_candidate_condition(sources, isochrone, c_delta=0.5, g_delta=1.5, colour='bp_rp',
                                  source_error_weight=3.0):
    """Returns a boolean array indicating which of the sources are labeled as candidates based on their
    magnitude and colour.

    Args:
        sources (Dataframe): Dataframe containing sources
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone
        c_delta (float): Maximum colour separation
        g_delta (float): Maximum magnitude separation
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        source_error_weight (float): How many source errors are added to the maximum separations

    Returns:
        passed_condition_indices (bool, array): Boolean array indicating which sources satisfied the condition.
    """
    f_c, f_g = isochrone_features_function(sources, isochrone, c_delta, g_delta, colour=colour, use_source_errors=True,
                                           source_error_weight=source_error_weight, scale_features=True)
    passed_condition_indices = np.linalg.norm(np.stack((f_c, f_g)).T, axis=1) < 1.0
    return passed_condition_indices


def candidate_conditions(sources, cluster):
    """Returns a boolean array indicating which of the sources are labeled as candidates based on their
    proper motion, parallax, magnitude and colour.

    Args:
        sources (Dataframe): Dataframe containing sources.
        cluster (Cluster): Cluster object containing the cluster properties.

    Returns:
        candidate_indices (array): Boolean array of the same size as the number of sources. True values indicate
            which sources are labeled as candidate.
    """

    f_pm_indices = pm_candidate_condition(sources, cluster.pmra, cluster.pmdec, cluster.delta_pm,
                                          cluster.source_error_weight)
    f_plx_indices = plx_candidate_condition(sources.loc[f_pm_indices], cluster.parallax, cluster.delta_plx_plus,
                                            cluster.delta_plx_min, cluster.source_error_weight)
    f_iso_indices = isochrone_candidate_condition(sources.loc[f_pm_indices][f_plx_indices], cluster.isochrone,
                                                  cluster.delta_c, cluster.delta_g, cluster.isochrone_colour,
                                                  cluster.source_error_weight)

    # Combine boolean arrays to find the sources which satisfied all three candidate conditions.
    candidate_indices = f_pm_indices.copy()
    candidate_indices[f_pm_indices] = f_plx_indices
    second_indices = f_pm_indices.copy()
    second_indices[f_pm_indices] = f_plx_indices
    candidate_indices[second_indices] = f_iso_indices

    return candidate_indices
