import pandas as pd


class Subset(pd.DataFrame):
    def __init__(self, sources, prob_threshold):
        super(Subset, self).__init__(sources)
        self.prob_threshold = prob_threshold

    def hp(self, min_prob=None, tidal_radius=None):
        if min_prob is None:
            min_prob = self.prob_threshold
        if tidal_radius is not None:
            prob_subset = self[(min_prob <= self['PMemb']) & (self['r'] <= tidal_radius)].copy()
            return prob_subset
        else:
            return self[min_prob <= self['PMemb']].copy()

    def lp(self, max_prob=None, tidal_radius=None):
        if max_prob is None:
            max_prob = self.prob_threshold
        if tidal_radius is not None:
            prob_subset = self[(max_prob >= self['PMemb']) | (self['r'] >= tidal_radius)].copy()
            return prob_subset
        else:
            return self[max_prob >= self['PMemb']].copy()


class Sources:
    def __init__(self, train_members, comparison_members, candidates, noise, prob_threshold=0.8,
                 train_ref='Cantat-Gaudin+18', comparison_ref='Tarricq+22'):
        self.train_members = Subset(train_members, prob_threshold)
        self.train_members_ref = train_ref
        self.comparison_members = Subset(comparison_members, prob_threshold)
        self.comparison_members_ref = comparison_ref
        self.candidates = Subset(candidates, prob_threshold)
        self.noise = Subset(noise, prob_threshold)
        self.prob_threshold = prob_threshold

    @property
    def all_sources(self):
        return pd.concat((self.candidates, self.noise))

    def add_field(self, f, label):
        for subset in [self.train_members, self.comparison_members, self.candidates, self.noise]:
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            subset[label] = subset.apply(f, axis=1, result_type=result_type)
