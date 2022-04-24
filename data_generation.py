import random
import torch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from data_filters import fields, pm_distance, plx_distance, isochrone_delta
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def normalize(dataframe, labels_not_normalized=None, all_data=None):
    if all_data is not None:
        data_norm = (dataframe - all_data.mean()) / all_data.std()
    else:
        data_norm = (dataframe - dataframe.mean()) / dataframe.std()
    if labels_not_normalized is not None:
        data_norm[labels_not_normalized] = dataframe[labels_not_normalized]
    return data_norm


def generate_rf_input(sources, train_fields, test_fraction=0.3, seed=42):
    members = sources.train_members.hp()
    noise = sources.noise

    n_members = len(members)
    n_noise = len(noise)

    np.random.seed(seed)
    member_test_indices = np.random.choice(np.arange(n_members), int(n_members * test_fraction), replace=False)
    member_train_indices = np.delete(np.arange(n_members), member_test_indices, axis=0)
    noise_test_indices = np.random.choice(np.arange(n_noise), int(n_noise * test_fraction), replace=False)
    noise_train_indices = np.delete(np.arange(n_noise), noise_test_indices, axis=0)

    train_members = members[member_train_indices][train_fields]
    test_members = members[member_test_indices][train_fields]

    train_noise = noise[noise_train_indices][train_fields]
    test_noise = noise[noise_test_indices][train_fields]

    x_train = pd.concat([train_members, train_noise], ignore_index=True).to_numpy()
    x_test = pd.concat([test_members, test_noise], ignore_index=True).to_numpy()

    y_train = np.concatenate((np.ones(len(member_train_indices)), np.zeros(len(noise_train_indices))))
    y_test = np.concatenate((np.ones(len(member_test_indices)), np.zeros(len(noise_test_indices))))

    return x_train, y_train, x_test, y_test


def generate_rf_candidate_samples(sources, train_fields, pm_dist, plx_dist, isochrone_del, n_samples):
    candidate_evaluation_set = []
    for _ in tqdm(range(n_samples), total=n_samples, desc="Generating candidate samples..."):
        sample_candidates_normalized = sample_candidate_data(sources, train_fields, pm_dist, plx_dist, isochrone_del,
                                                             sample_photometric=True)
        print(sample_candidates_normalized.shape)
        candidate_evaluation_set.append(sample_candidates_normalized)

    return candidate_evaluation_set


class OSLDataset(Dataset):
    def __init__(self, sources, train_fields, test_fraction, max_members=2000, max_noise=10000, min_support_size=5,
                 max_support_size=50, test=True, seed=42):
        members = sources.train_members.hp()
        noise = sources.noise

        self.n_members = len(members)
        self.n_noise = len(noise)
        self.test_size = int(self.n_members * test_fraction)

        np.random.seed(seed)
        member_test_indices = np.random.choice(np.arange(self.n_members), int(self.n_members * test_fraction), replace=False)
        member_train_indices = np.delete(np.arange(self.n_members), member_test_indices, axis=0)

        noise_test_indices = np.random.choice(np.arange(self.n_noise), int(self.n_noise * test_fraction), replace=False)
        noise_train_indices = np.delete(np.arange(self.n_noise), noise_test_indices, axis=0)

        if test:
            member_sample_size = int(max_members * test_fraction)
            noise_sample_size = int(max_noise * test_fraction)
            member_indices = np.random.choice(member_test_indices, member_sample_size, replace=True)
            noise_indices = np.random.choice(noise_test_indices, noise_sample_size, replace=True)
        else:
            member_sample_size = max_members - int(max_members * test_fraction)
            noise_sample_size = max_noise - int(max_noise * test_fraction)
            member_indices = np.random.choice(member_train_indices, member_sample_size, replace=True)
            noise_indices = np.random.choice(noise_train_indices, noise_sample_size, replace=True)

        members = normalize(members[train_fields], all_data=sources.all_sources[train_fields]).to_numpy()
        noise = normalize(noise[train_fields], all_data=sources.all_sources[train_fields]).to_numpy()

        self.dataset = []

        member_indices_2 = list(np.arange(self.n_members))
        max_support_size = min(int(self.n_members - 1), max_support_size)
        for source_to_classify_idx in member_indices:
            source_to_classify = members[source_to_classify_idx]

            size_support_set = np.random.randint(min_support_size, max_support_size)
            support_set_indices = np.random.choice(np.delete(member_indices_2, source_to_classify_idx, axis=0),
                                                   size_support_set, replace=False)
            support_set = members[support_set_indices]

            nn_input = np.concatenate((support_set, np.tile(source_to_classify, (size_support_set, 1))), axis=1)
            self.dataset.append((nn_input, [0, 1]))

        max_support_size = min(int(self.n_members), max_support_size)
        for source_to_classify_idx in noise_indices:
            source_to_classify = noise[source_to_classify_idx]

            size_support_set = np.random.randint(min_support_size, max_support_size)
            support_set_indices = np.random.choice(member_indices_2, size_support_set, replace=False)
            support_set = members[support_set_indices]

            nn_input = np.concatenate((support_set, np.tile(source_to_classify, (size_support_set, 1))), axis=1)
            self.dataset.append((nn_input, [1, 0]))

        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class OSLCandidateDataset(Dataset):
    def __init__(self, candidates, members, min_support_size, max_support_size):
        self.dataset = []
        member_indices = list(np.arange(len(members)))
        for source_to_classify in candidates:
            size_support_set = np.random.randint(min_support_size, max_support_size)
            support_indices = np.random.choice(member_indices, size_support_set, replace=False)
            support_set = members[support_indices]
            nn_input = np.concatenate((support_set, np.tile(source_to_classify, (size_support_set, 1))), axis=1)

            self.dataset.append(nn_input)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x = self.dataset[item]
        return torch.FloatTensor(x)


# def flux_to_mag(flux, band):
#     g_mag_correction = 25.68836574676364
#     bp_mag_correction = 25.351388202706424
#     rp_mag_correction = 24.761920004514547
#     uncorrected_mag = -2.5 * np.log10(flux)
#     if band == 'g':
#         return uncorrected_mag + g_mag_correction
#     elif band == 'bp':
#         return uncorrected_mag + bp_mag_correction
#     elif band == 'rp':
#         return uncorrected_mag + rp_mag_correction
#     else:
#         raise ValueError(f'band {band} is unknown, use g/bp/rp')


def sample_candidate_data(sources, train_fields, pm_dist, plx_dist, isochrone_del, sample_photometric=False):
    sample_df = sources.candidates.copy().dropna()

    n_sources = sample_df.shape[0]
    astrometric_dim = len(fields['astrometric'])

    astrometric_means = np.zeros((astrometric_dim, n_sources))
    for i in range(astrometric_dim):
        astrometric_means[i] = sample_df[fields['astrometric'][i]]

    def get_corr_idx(row, col):
        return 5 * min(row, col) + max(row, col) - int(np.sum(np.arange(min(row, col) + 2)))

    astrometric_cov_matrices = np.zeros((astrometric_dim, astrometric_dim, n_sources))
    for i in range(astrometric_dim):
        for j in range(i, astrometric_dim):
            if i == j:
                sigma = sample_df[fields['astrometric_error'][i]].to_numpy()
                astrometric_cov_matrices[i, j] = sigma**2
            else:
                corr_name_idx = get_corr_idx(i, j)
                corr = sample_df[fields['astrometric_corr'][corr_name_idx]].to_numpy()
                sigma_i = sample_df[fields['astrometric_error'][i]].to_numpy()
                sigma_j = sample_df[fields['astrometric_error'][j]].to_numpy()
                cov = corr * sigma_i * sigma_j
                astrometric_cov_matrices[i, j] = cov
                astrometric_cov_matrices[j, i] = cov

    samples = np.zeros((n_sources, astrometric_dim + 2))
    for i in range(n_sources):
        astro_mn = multivariate_normal(mean=astrometric_means[:, i],
                                       cov=astrometric_cov_matrices[:, :, i])
        samples[i, :5] = astro_mn.rvs(1)

    sample_df[fields['astrometric']] = samples[:, :5]

    if sample_photometric:
        photometric_dim = len(fields['photometric'])
        photometric_cov_matrices = np.zeros((photometric_dim, n_sources))

        photometric_means = np.zeros((photometric_dim, n_sources))
        for i in range(photometric_dim):
            photometric_means[i] = sample_df[fields['photometric'][i]]

        for i in range(photometric_dim):
            sigma = sample_df[fields['photometric_error'][i]].to_numpy()
            photometric_cov_matrices[i] = sigma ** 2

        for i in range(n_sources):
            photometric_mn = multivariate_normal(mean=photometric_means[:, i],
                                                 cov=photometric_cov_matrices[:, i])
            samples[i, 5:7] = photometric_mn.rvs(1)

        sample_df[fields['photometric']] = samples[:, 5:7]

    sample_df['pm_d'] = sample_df.apply(pm_dist, axis=1)
    sample_df['plx_d'] = sample_df.apply(plx_dist, axis=1)
    sample_df[['bp_rp_d', 'gmag_d']] = sample_df.apply(isochrone_del, axis=1, result_type='expand')

    sample_candidates = normalize(sample_df[train_fields], all_data=sources.all_sources[train_fields]).to_numpy()

    return sample_candidates


def generate_osl_candidate_samples(sources, cluster, isochrone, candidate_filter_kwargs, train_fields, n_samples,
                                   min_support_size=5, max_support_size=50):
    members = normalize(sources.train_members.hp()[train_fields], all_data=sources.all_sources[train_fields]).to_numpy()

    max_support_size = min(len(members), max_support_size)
    candidate_evaluation_set = []

    pm_dist = pm_distance(cluster)
    plx_dist = plx_distance(cluster)
    isochrone_del = isochrone_delta(isochrone, candidate_filter_kwargs)

    for _ in tqdm(range(n_samples), total=n_samples, desc="Generating candidate samples..."):
        candidate_sample = sample_candidate_data(sources, train_fields, pm_dist, plx_dist, isochrone_del,
                                                 sample_photometric=True)

        candidate_evaluation_set.append(DataLoader(OSLCandidateDataset(candidate_sample, members, min_support_size,
                                                                       max_support_size)))

    return candidate_evaluation_set
