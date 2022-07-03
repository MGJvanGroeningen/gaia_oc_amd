import pandas as pd

from gaia_oc_amd.data_preparation.sets import member_set, candidate_and_non_members_set
from gaia_oc_amd.data_preparation.features import phot_g_mean_mag_error_function, bp_rp_error_function, \
    radius_feature_function, pm_feature_function, plx_feature_function, isochrone_features_function, add_features


def parse_cone(cluster, cone, isochrone, member_ids, member_probs=None, comparison_ids=None, comparison_probs=None,
               max_r=60., pm_errors=5., g_delta=1.5, bp_rp_delta=0.5, source_errors=3.):
    sources = []
    comparison = None

    members = member_set(cone, member_ids, member_probs)
    sources.append(members)
    if comparison_ids is not None:
        comparison = member_set(cone, comparison_ids, comparison_probs)
        sources.append(comparison)

    cone = cone[~cone['source_id'].isin(member_ids)].copy()
    sources.append(cone)

    cluster.update_parameters(members)
    cluster.set_feature_parameters(members, max_r, pm_errors, g_delta, bp_rp_delta, source_errors)

    feature_functions = [phot_g_mean_mag_error_function(), bp_rp_error_function(), radius_feature_function(cluster),
                         pm_feature_function(cluster), plx_feature_function(cluster),
                         isochrone_features_function(cluster, isochrone)]
    feature_labels = ['phot_g_mean_mag_error', 'bp_rp_error', 'f_r', 'f_pm', 'f_plx', ['f_c', 'f_g']]
    add_features(sources, feature_functions, feature_labels)

    candidates, non_members = candidate_and_non_members_set(cone, cluster, isochrone)
    candidates = pd.concat((candidates, members))

    return members, candidates, non_members, comparison
