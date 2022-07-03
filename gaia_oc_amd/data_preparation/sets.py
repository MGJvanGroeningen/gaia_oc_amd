import pandas as pd

from gaia_oc_amd.data_preparation.candidate_selection import candidate_conditions


class Subset(pd.DataFrame):
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
    def __init__(self, members, candidates, non_members, comparison_members=None,
                 members_label='train', comparison_label='comparison'):
        self.members = Subset(members)
        self.candidates = Subset(candidates)
        self.non_members = Subset(non_members)

        if comparison_members is not None:
            self.comparison_members = Subset(comparison_members)
        else:
            self.comparison_members = None

        self.members_label = members_label
        self.comparison_label = comparison_label

    @property
    def all_sources(self):
        return pd.concat((self.candidates, self.non_members))

    def add_feature(self, f, label):
        for subset in [self.members, self.comparison_members, self.candidates, self.non_members]:
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            subset[label] = subset.apply(f, axis=1, result_type=result_type)


def candidate_and_non_members_set(cone, cluster, isochrone):
    conditions = candidate_conditions(cluster, isochrone)
    candidate_indices = cone.apply(conditions, axis=1)

    candidates = cone[candidate_indices].copy()
    non_members = cone[~candidate_indices].copy()
    return candidates, non_members


def member_set(cone, member_ids, member_probs=None):
    members = cone[cone['source_id'].isin(member_ids)].copy()
    if member_probs is not None:
        member_ids_and_probs = pd.DataFrame({'source_id': member_ids, 'PMemb': member_probs})
        members = pd.merge(member_ids_and_probs, members, on='source_id', how='inner', suffixes=('', '_y'), copy=False)
        members = members.drop(['PMemb_y'], axis=1)
    return members
