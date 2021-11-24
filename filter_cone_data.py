import numpy as np
import pandas as pd
import random
import os


def normalize(dataframe, labels_not_normalized=None):
    data_norm = (dataframe - dataframe.mean()) / dataframe.std()
    if labels_not_normalized is not None:
        data_norm[labels_not_normalized] = dataframe[labels_not_normalized]
    return data_norm


def load_cone_data(data_dir, cone_file):
    practice_data_cone_file = os.path.join(data_dir, cone_file)

    # create cone dataframe
    cone_df = pd.read_csv(practice_data_cone_file)
    return cone_df


def load_cluster_data(data_dir, cluster_file):
    practice_data_cluster_file = os.path.join(data_dir, cluster_file)

    # create cluster dataframe
    cluster_df = pd.read_csv(practice_data_cluster_file, sep='\t', header=61)
    cluster_df = cluster_df.iloc[2:]
    cluster_df = cluster_df.reset_index(drop=True)
    return cluster_df


def isochrone_distance_rule(df, max_distance_bp_rp, max_distance_gmag):
    bp_rps = df['bp_rp'].values
    gmags = df['phot_g_mean_mag'].values

    members = np.array([bp_rps, gmags]).T
    scale_factors = np.array([1 / max_distance_bp_rp, 1 / max_distance_gmag])

    def candidate_rule(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.any(distances < 1):
            candidate = True
        return candidate

    return candidate_rule


def pm_distance_rule(df, max_distance_pmra, max_distance_pmdec):
    pmras = df['pmra'].values
    pmdecs = df['pmdec'].values

    members = np.array([pmras, pmdecs]).T
    scale_factors = np.array([1 / max_distance_pmra, 1 / max_distance_pmdec])

    def candidate_rule(row):
        pmra, pmdec = row['pmra'], row['pmdec']
        point = np.array([pmra, pmdec])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.mean(distances) < 1:
            candidate = True
        return candidate

    return candidate_rule


def plx_distance_rule(df, max_distance_gmag, plx_sigma):
    gamgs = df['phot_g_mean_mag'].values
    plxs = df['parallax'].values
    plx_errors = df['parallax_error'].values

    members = np.array([gamgs, plxs]).T
    max_delta_squared = np.array([np.tile(np.array([max_distance_gmag]), len(plx_errors)),
                                  (plx_sigma * plx_errors)**2]).T

    def plx_candidate_rule(row):
        gmag, plx, plx_error = row['phot_g_mean_mag'], row['parallax'], row['parallax_error']
        point = np.array([gmag, plx])
        deltas_squared = (members - point)**2

        candidate = False
        if np.any(np.all(deltas_squared < max_delta_squared + np.array([0., (plx_sigma * plx_error)**2]), axis=1)):
            candidate = True
        return candidate

    return plx_candidate_rule


def make_candidate_rule(member_df, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d):
    plx_candidate_rule = plx_distance_rule(member_df, gmag_max_d, plx_sigma)
    pm_candidate_rule = pm_distance_rule(member_df, pm_max_d, pm_max_d)
    isochrone_candidate_rule = isochrone_distance_rule(member_df, bp_rp_max_d, gmag_max_d)

    def candidate_rule(row):
        candidate = False
        if plx_candidate_rule(row) and pm_candidate_rule(row) and isochrone_candidate_rule(row):
            candidate = True
        return candidate

    return candidate_rule


def make_high_prob_member_df(cone_df, cluster_df, candidate_selection_columns, probability_threshold):
    high_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >=
                                              probability_threshold)]['Source'].astype(np.int64).values
    high_prob_member_df = cone_df[(cone_df['source_id'].isin(high_prob_member_source_ids))][
        candidate_selection_columns].dropna()
    return high_prob_member_df


def make_low_prob_member_df(cone_df, cluster_df, candidate_selection_columns, probability_threshold):
    low_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) <
                                             probability_threshold)]['Source'].astype(np.int64).values
    low_prob_member_df = cone_df[(cone_df['source_id'].isin(low_prob_member_source_ids))][
        candidate_selection_columns].dropna()
    return low_prob_member_df


def make_field_df(cone_df, high_prob_member_df, candidate_selection_columns):
    field_df = cone_df.drop(high_prob_member_df.index)[candidate_selection_columns].dropna()
    return field_df


def make_non_member_df(cone_df,
                       member_df,
                       candidate_selection_columns,
                       candidate_rule):
    field_df = make_field_df(cone_df, member_df, candidate_selection_columns)

    def not_candidate(row):
        return not candidate_rule(row)

    non_member_df = field_df[field_df.apply(not_candidate, axis=1)]

    return non_member_df


def generate_training_data(data_dir,
                           cone_file,
                           cluster_file,
                           probability_threshold,
                           candidate_selection_columns,
                           plx_sigma,
                           gmag_max_d,
                           pm_max_d,
                           bp_rp_max_d):
    cone_df = load_cone_data(data_dir, cone_file)
    cluster_df = load_cluster_data(data_dir, cluster_file)

    cluster = make_high_prob_member_df(cone_df, cluster_df, candidate_selection_columns, probability_threshold)
    cluster['member'] = 1

    combined_rule = make_candidate_rule(cluster, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d)
    noise = make_non_member_df(cone_df, cluster, candidate_selection_columns, combined_rule)
    noise['member'] = 0

    train = []

    n_cluster_sources = len(cluster)
    n_noise_sources = len(noise)

    print(n_cluster_sources, n_noise_sources)

    # concat everything to compute ang coord and normalize
    all_data = pd.concat((cluster, noise), sort=False, ignore_index=True)

    all_data_normlz = normalize(all_data, ['member'])

    cluster_normalized = all_data_normlz[all_data_normlz["member"] == 1][candidate_selection_columns].to_numpy()
    noise_normalized = all_data_normlz[all_data_normlz["member"] == 0][candidate_selection_columns].to_numpy()

    min_support_size = 5
    max_support_size = 50

    for neg_ex in range(n_cluster_sources):
        size_support_set = random.randint(min_support_size, min(int(n_cluster_sources), max_support_size))

        index_support = random.sample(list(np.arange(n_cluster_sources)), size_support_set)
        reference_points = cluster_normalized[index_support, :].copy()
        train.append((noise_normalized[neg_ex], 0, reference_points.copy()))

    for pos_ex in range(n_noise_sources):
        idx_pos = random.randint(0, n_cluster_sources - 1)

        size_support_set = random.randint(min_support_size, min(int(n_cluster_sources), max_support_size))

        list_of_indexes = list(np.arange(n_cluster_sources))
        list_of_indexes.remove(idx_pos)

        index_support = random.sample(list_of_indexes, size_support_set)
        reference_points = cluster_normalized[index_support, :].copy()
        train.append((cluster_normalized[idx_pos], 1, reference_points.copy()))

    return train


def generate_candidate_data(data_dir,
                            cone_file,
                            cluster_file,
                            probability_threshold,
                            candidate_selection_columns,
                            plx_sigma,
                            gmag_max_d,
                            pm_max_d,
                            bp_rp_max_d):
    cone_df = load_cone_data(data_dir, cone_file)
    cluster_df = load_cluster_data(data_dir, cluster_file)

    cluster = make_high_prob_member_df(cone_df, cluster_df, candidate_selection_columns, probability_threshold)
    cluster['candidate'] = 0

    field = make_field_df(cone_df, cluster, candidate_selection_columns)

    candidate_rule = make_candidate_rule(cluster, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d)
    candidates = field[field.apply(candidate_rule, axis=1)]
    candidates['candidate'] = 1

    n_cluster_sources = len(cluster)
    n_candidate_sources = len(candidates)

    print(n_candidate_sources)

    all_data = pd.concat((cluster, candidates), sort=False, ignore_index=True)

    all_data_normlz = normalize(all_data, ['candidate'])

    cluster_normalized = all_data_normlz[all_data_normlz["candidate"] == 0][candidate_selection_columns].to_numpy()
    candidate_normalized = all_data_normlz[all_data_normlz["candidate"] == 1][candidate_selection_columns].to_numpy()

    min_support_size = 5
    max_support_size = 50

    candidate_set = []

    for candidate in range(n_candidate_sources):
        size_support_set = random.randint(min_support_size, min(int(n_cluster_sources), max_support_size))

        index_support = random.sample(list(np.arange(n_cluster_sources)), size_support_set)
        reference_points = cluster_normalized[index_support, :].copy()
        candidate_set.append((candidate_normalized[candidate], reference_points.copy()))

    return candidate_set


def divide_cone(cone_df,
                cluster_df,
                candidate_selection_columns,
                probability_threshold,
                plx_sigma,
                gmag_max_d,
                pm_max_d,
                bp_rp_max_d,
                drop_db_candidates=False):
    n_sources = len(cone_df)
    print(f'Total number of sources in cone: {n_sources}')

    # select high probability members from the cone
    high_prob_member_df = make_high_prob_member_df(cone_df,
                                                   cluster_df,
                                                   candidate_selection_columns,
                                                   probability_threshold)
    n_high_prob_member = len(high_prob_member_df)
    print(f'{n_high_prob_member} cluster sources were selected with a membership probability of >= '
          f'{probability_threshold}')
    
    # select low probability members from the cone
    low_prob_member_df = make_low_prob_member_df(cone_df,
                                                 cluster_df,
                                                 candidate_selection_columns,
                                                 probability_threshold)
    n_low_prob_members = len(low_prob_member_df)
    print(f'{n_low_prob_members} cluster sources had a too low probability to be used for the training set')
    
    # drop high probability members from the cone to obtain the field
    field_df = make_field_df(cone_df,
                             high_prob_member_df,
                             candidate_selection_columns)
    n_field = len(field_df)
    n_incomplete_rows = n_sources - n_field - n_high_prob_member
    print(f'{n_incomplete_rows} sources were removed from the cone because of incomplete data')

    # optionally prevent the low probability members to be selected as candidates
    if drop_db_candidates:
        suffix = '(low database probability cluster sources were excluded)'
        field_df = field_df.drop(low_prob_member_df.index)
    else:
        suffix = '(may include low database probability cluster sources)'

    # select member candidates based on distance from high probability members in plx, pm and isochrone space
    candidate_rule = make_candidate_rule(high_prob_member_df, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d)
    candidate_df = field_df[field_df.apply(candidate_rule, axis=1)]
    n_candidates = len(candidate_df)
    print(f'{n_candidates} sources were selected as candidates {suffix}')

    # select non members by removing the candidates from the field
    if len(candidate_df) > 0:
        non_member_df = field_df.append(candidate_df).drop_duplicates(keep=False)
    else:
        non_member_df = field_df
    n_non_members = len(non_member_df)
    print(f'{n_non_members} sources were selected as non members')

    return high_prob_member_df, low_prob_member_df, candidate_df, non_member_df
