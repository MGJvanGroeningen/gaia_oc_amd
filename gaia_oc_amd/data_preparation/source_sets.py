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
    if member_probs is not None:
        member_ids_and_probs = pd.DataFrame({'source_id': member_ids, 'PMemb': member_probs})
        members = pd.merge(member_ids_and_probs, cone, on='source_id', how='inner', suffixes=('', '_y'), copy=False)
        if 'PMemb_y' in members.columns.to_list():
            members = members.drop(['PMemb_y'], axis=1)
    else:
        members = cone[cone['source_id'].isin(member_ids)].copy()
        members['PMemb'] = 1.0
    return members


def get_duplicate_sources(sources1, sources2, keep='first'):
    """Function that returns the sources that are in the first set which are also in the second set or vice versa.

    Args:
        sources1 (Dataframe): Dataframe containing the data of sources
        sources2 (Dataframe): Another dataframe containing the data of sources
        keep (str): Whether to keep the sources from the first or the second (last) set. Use either 'first' or 'last'.

    Returns:
        dup_sources (Dataframe): Dataframe containing the sources which are in both sets

    """
    if keep == 'first':
        concat_sources = pd.concat((sources2, sources1))
    elif keep == 'last':
        concat_sources = pd.concat((sources1, sources2))
    else:
        raise ValueError("The 'keep' argument must be either 'first' or 'last'.")
    dup_sources = concat_sources[concat_sources.duplicated('source_id')]
    return dup_sources
