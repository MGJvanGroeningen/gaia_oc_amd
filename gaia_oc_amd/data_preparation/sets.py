import pandas as pd

from gaia_oc_amd.data_preparation.candidate_selection import candidate_conditions


class Subset(pd.DataFrame):
    """A pandas Dataframe class with additional functions to select high probability and low probability members.

    Args:
        sources (Dataframe): A dataframe containing source data

    """
    def __init__(self, sources):
        super(Subset, self).__init__(sources)

    def hp(self, min_prob=0.0, tidal_radius=None):
        if tidal_radius is not None:
            return self[(min_prob <= self['PMemb']) & (self['f_r'] <= tidal_radius)].copy()
        else:
            return self[min_prob <= self['PMemb']].copy()

    def lp(self, max_prob=1.0, tidal_radius=None):
        if tidal_radius is not None:
            return self[(max_prob > self['PMemb']) | (self['f_r'] > tidal_radius)].copy()
        else:
            return self[max_prob > self['PMemb']].copy()


class Sources:
    """A class for storing the different source sets in one place. This class is expected by most plotting functions.

    Args:
        members (Dataframe): Dataframe containing member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources
        comparison (Dataframe): Dataframe containing comparison sources

    """
    def __init__(self, members, candidates, non_members, comparison=None,
                 members_label='train', comparison_label='comparison'):
        self.members = Subset(members)
        self.candidates = Subset(candidates)
        self.non_members = Subset(non_members)

        if comparison is not None:
            self.comparison = Subset(comparison)
        else:
            self.comparison = None

        self.members_label = members_label
        self.comparison_label = comparison_label

    @property
    def all_sources(self):
        return pd.concat((self.candidates, self.non_members))


def candidate_and_non_members_set(cone, cluster, isochrone):
    """Divides the cone sources into either candidates or non-members, using a number of candidate conditions.

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.
        cluster (Cluster): Cluster object
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.

    Returns:
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources

    """
    conditions = candidate_conditions(cluster, isochrone)
    candidate_indices = cone.apply(conditions, axis=1)

    candidates = cone[candidate_indices].copy()
    non_members = cone[~candidate_indices].copy()
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
