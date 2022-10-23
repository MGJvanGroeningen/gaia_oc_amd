import pandas as pd

from gaia_oc_amd.data_preparation.candidate_selection import candidate_conditions


def candidate_and_non_member_set(cone, cluster):
    """Divides the cone sources into either candidates or non-members, using a number of candidate conditions.

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.
        cluster (Cluster): Cluster object

    Returns:
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources

    """
    conditions = candidate_conditions(cluster)
    candidate_indices = cone.apply(conditions, axis=1)

    candidates = cone[candidate_indices].copy().reset_index(drop=True)
    non_members = cone[~candidate_indices].copy().reset_index(drop=True)

    return candidates, non_members


def member_set(cone, member_ids, member_probs=None):
    """Selects the member sources in the cone with the member source identities.
    Optionally also assigns their probabilities.

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.
        member_ids (Series, array): Source identities of the members
        member_probs (Series, array): Membership probabilities of the members

    Returns:
        members (Dataframe): Dataframe containing member sources

    """
    members = cone[cone['source_id'].isin(member_ids)].copy()
    if member_probs is not None:
        member_ids_and_probs = pd.DataFrame({'source_id': member_ids, 'PMemb': member_probs})
        members = pd.merge(member_ids_and_probs, members, on='source_id', how='inner', suffixes=('', '_y'), copy=False)
        members = members.drop(['PMemb_y'], axis=1)
    return members
