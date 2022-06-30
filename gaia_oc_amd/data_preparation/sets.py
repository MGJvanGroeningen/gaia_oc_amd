import pandas as pd
import numpy as np
import time

from gaia_oc_amd.data_preparation.candidate_selection import candidate_filter


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
    print('Finding candidates...', end=' ')
    t0 = time.time()

    can_filter = candidate_filter(cluster, isochrone)
    candidate_indices = cone.apply(can_filter, axis=1)

    candidates = cone[candidate_indices].copy()
    non_members = cone[~candidate_indices].copy()

    print(f'done in {np.round(time.time() - t0, 1)} sec')
    return candidates, non_members


def member_set(cone, ids, probs=None):
    sources = cone[cone['source_id'].isin(ids)].copy()
    if probs is not None:
        ids_and_probs = pd.DataFrame({'source_id': ids, 'PMemb': probs})
        sources = pd.merge(ids_and_probs, sources, on='source_id', how='inner', suffixes=('', '_y'), copy=False)
        sources = sources.drop(['PMemb_y'], axis=1)
    return sources
