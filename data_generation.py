import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from data_filters import fields
from tqdm import tqdm


def normalize(dataframe, labels_not_normalized=None, all_data=None):
    if all_data is not None:
        data_norm = (dataframe - all_data.mean()) / all_data.std()
    else:
        data_norm = (dataframe - dataframe.mean()) / dataframe.std()
    if labels_not_normalized is not None:
        data_norm[labels_not_normalized] = dataframe[labels_not_normalized]
    return data_norm


def generate_rf_input(members, noise, candidates, train_fields, test_fraction=0.3):
    n_train_members, n_test_members = label_test_sources(members, test_fraction)
    n_train_noise, n_test_noise = label_test_sources(noise, test_fraction)

    train_members = members[members['test'] == 0][train_fields]
    test_members = members[members['test'] == 1][train_fields]

    train_noise = noise[noise['test'] == 0][train_fields]
    test_noise = noise[noise['test'] == 1][train_fields]

    x_train = pd.concat([train_members, train_noise], ignore_index=True).to_numpy()
    x_test = pd.concat([test_members, test_noise], ignore_index=True).to_numpy()

    y_train = np.concatenate((np.ones(n_train_members), np.zeros(n_train_noise)))
    y_test = np.concatenate((np.ones(n_test_members), np.zeros(n_test_noise)))

    print(f'\nTraining set size: {n_train_members + n_train_noise}')
    print(f'\nTest set size: {n_test_members + n_test_noise}')

    # create test data for random forest
    x_eval = candidates[train_fields].values
    n_eval = x_eval.shape[0]
    print(f'Eval set size: {n_eval}')

    return x_train, y_train, x_test, y_test, x_eval


def label_test_sources(dataset, test_fraction):
    dataset['test'] = 0
    n_sources = len(dataset)
    n_test_sources = int(test_fraction * n_sources)
    n_train_sources = n_sources - n_test_sources
    test_indices = np.random.choice(n_sources, n_test_sources, replace=False)
    dataset.iloc[test_indices, dataset.columns.get_loc('test')] = 1
    return n_train_sources, n_test_sources


def append_neg_example(data_list, neg_idx, min_support_size, max_support_size, n_member_sources, members, noise,
                       train_on_pmemb):
    size_support_set = np.random.randint(min_support_size, min(int(n_member_sources), max_support_size))

    index_support = np.random.choice(list(np.arange(n_member_sources)), size_support_set, replace=False)
    if train_on_pmemb:
        reference_points = members[index_support, :].copy()
        data_list.append((np.concatenate((noise[neg_idx], np.array([1.]))), 0, reference_points.copy()))
    else:
        reference_points = members[index_support, :-1].copy()
        data_list.append((noise[neg_idx], 0, reference_points.copy()))


def append_pos_example(data_list, min_support_size, max_support_size, n_member_sources, members, member_probs,
                       train_on_pmemb):
    idx_pos = np.random.randint(0, n_member_sources - 1)

    size_support_set = np.random.randint(min_support_size, min(int(n_member_sources - 1), max_support_size))

    list_of_indexes = list(np.arange(n_member_sources))
    list_of_indexes.remove(idx_pos)

    index_support = np.random.choice(list_of_indexes, size_support_set, replace=False)
    if train_on_pmemb:
        reference_points = members[index_support, :].copy()
        data_list.append((np.concatenate((members[idx_pos, :-1], np.array([1.]))), member_probs[idx_pos],
                          reference_points.copy()))
    else:
        reference_points = members[index_support, :-1].copy()
        data_list.append((members[idx_pos, :-1], member_probs[idx_pos], reference_points.copy()))


def generate_nn_input(members, noise, candidates, train_fields, test_fraction, max_members, train_on_pmemb=True,
                      min_support_size=5, max_support_size=50):
    all_sources = pd.concat((members, noise, candidates), sort=False, ignore_index=True)

    members_normalized = normalize(members, labels_not_normalized=['PMemb'], all_data=all_sources)
    noise_normalized = normalize(noise, labels_not_normalized=['PMemb'], all_data=all_sources)

    n_pos_train_samples = int(max_members * (1 - test_fraction))
    n_pos_test_samples = int(max_members * test_fraction)

    n_train_members, n_test_members = label_test_sources(members_normalized, test_fraction)
    n_train_noise, n_test_noise = label_test_sources(noise_normalized, test_fraction)

    train_members = members_normalized[members_normalized['test'] == 0]
    test_members = members_normalized[members_normalized['test'] == 1]

    train_member_probs = train_members['PMemb'].to_numpy()
    trest_member_probs = test_members['PMemb'].to_numpy()

    train_members = train_members[train_fields + ['PMemb']].to_numpy()
    test_members = test_members[train_fields + ['PMemb']].to_numpy()

    train_noise = noise_normalized[noise_normalized['test'] == 0][train_fields].to_numpy()
    test_noise = noise_normalized[noise_normalized['test'] == 1][train_fields].to_numpy()

    train = []
    test = []

    for neg_train_ex in range(n_train_noise):
        append_neg_example(train, neg_train_ex, min_support_size, max_support_size, n_train_members,
                           train_members, train_noise, train_on_pmemb)

    for neg_test_ex in range(n_test_noise):
        append_neg_example(test, neg_test_ex, min_support_size, max_support_size, n_test_members,
                           test_members, test_noise, train_on_pmemb)

    for _ in range(n_pos_train_samples):
        append_pos_example(train, min_support_size, max_support_size, n_train_members,
                           train_members, train_member_probs, train_on_pmemb)

    for _ in range(n_pos_test_samples):
        append_pos_example(test, min_support_size, max_support_size, n_test_members,
                           test_members, trest_member_probs, train_on_pmemb)

    return train, test


def flux_to_mag(flux, band):
    g_mag_correction = 25.68836574676364
    bp_mag_correction = 25.351388202706424
    rp_mag_correction = 24.761920004514547
    uncorrected_mag = -2.5 * np.log10(flux)
    if band == 'g':
        return uncorrected_mag + g_mag_correction
    elif band == 'bp':
        return uncorrected_mag + bp_mag_correction
    elif band == 'rp':
        return uncorrected_mag + rp_mag_correction
    else:
        raise ValueError(f'band {band} is unknown, use g/bp/rp')


def sample_candidate_data(df, sample_photometric=False):
    sample_df = df.copy().dropna()

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
        # THIS CURRENTLY DOES NOT WORK PROPERLY
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
            photometric_sample = photometric_mn.rvs(1)
            g_mag = flux_to_mag(photometric_sample[0:1], 'g')
            bp_rp = flux_to_mag(photometric_sample[1:2], 'bp') - flux_to_mag(photometric_sample[2:3], 'rp')

            samples[i] = np.append((samples[i], g_mag, bp_rp), axis=0)

        sample_df['phot_g_mean_mag'] = samples[:, 5]
        sample_df['bp_rp'] = samples[:, 6]

    return sample_df


def generate_candidate_samples(members, noise, candidates, train_fields, n_samples, train_on_pmemb,
                               min_support_size=5, max_support_size=50):
    all_sources = pd.concat((members, noise, candidates), sort=False, ignore_index=True)

    members_normalized = normalize(members, labels_not_normalized=['PMemb'],
                                   all_data=all_sources)[train_fields + ['PMemb']].to_numpy()

    n_member_sources = len(members)
    n_candidate_sources = len(candidates)

    candidate_set = [[] for _ in range(n_samples)]

    for sample_idx in tqdm(range(n_samples), total=n_samples, desc="Generating candidate samples..."):
        sample_candidates = sample_candidate_data(candidates)
        sample_candidates_normalized = normalize(sample_candidates, ['source_id'],
                                                 all_data=all_sources)[train_fields].to_numpy()

        for candidate_idx in range(n_candidate_sources):
            size_support_set = np.random.randint(min_support_size,
                                                 min(int(n_member_sources), max_support_size))
            index_support = np.random.choice(list(np.arange(n_member_sources)),
                                             size_support_set, replace=False)
            if train_on_pmemb:
                reference_points = members_normalized[index_support, :].copy()
                candidate_set[sample_idx].append((np.concatenate((sample_candidates_normalized[candidate_idx],
                                                                  np.array([1.]))),
                                                  reference_points.copy()))
            else:
                reference_points = members_normalized[index_support, :-1].copy()
                candidate_set[sample_idx].append((sample_candidates_normalized[candidate_idx], reference_points.copy()))

    return candidate_set
