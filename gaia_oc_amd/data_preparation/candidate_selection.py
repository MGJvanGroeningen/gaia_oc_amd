from gaia_oc_amd.data_preparation.features import pm_distance, plx_distance, isochrone_delta
from gaia_oc_amd.data_preparation.utils import norm


def pm_candidate_filter(cluster):
    pm_dist = pm_distance(cluster, use_source_errors=True)

    def can_filter(row):
        pm_d = pm_dist(row)

        candidate = False
        if pm_d < 1.0:
            candidate = True
        return candidate

    return can_filter


def plx_candidate_filter(cluster):
    plx_dist = plx_distance(cluster, use_source_errors=True)

    def can_filter(row):
        plx_d = plx_dist(row)

        candidate = False
        if plx_d < 1.0:
            candidate = True
        return candidate

    return can_filter


def isochrone_candidate_filter(cluster, isochrone):
    isochrone_del = isochrone_delta(cluster, isochrone, use_source_errors=True)

    def can_filter(row):
        isochrone_d = isochrone_del(row)

        candidate = False
        if norm(isochrone_d) < 1.0:
            candidate = True
        return candidate

    return can_filter


def candidate_filter(cluster, isochrone):
    filters = [plx_candidate_filter(cluster),
               pm_candidate_filter(cluster),
               isochrone_candidate_filter(cluster, isochrone)]

    def can_filter(row):
        candidate = True
        for filter_rule in filters:
            passed_filter = filter_rule(row)
            if not passed_filter:
                candidate = False
                break
        return candidate

    return can_filter
