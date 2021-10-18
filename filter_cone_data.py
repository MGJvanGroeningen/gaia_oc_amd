import numpy as np


def isochrone_distance_rule(df, max_distance_x, max_distance_y, x_name, y_name):
    xs = df[x_name].values
    ys = df[y_name].values

    members = np.array([xs, ys]).T
    scale_factors = np.array([1 / max_distance_x, 1 / max_distance_y])

    def candidate_rule(x, y):
        point = np.array([x, y])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.any(distances < 1):
            candidate = True
        return candidate

    return candidate_rule


def pm_distance_rule(df, max_distance_x, max_distance_y, x_name, y_name):
    xs = df[x_name].values
    ys = df[y_name].values

    members = np.array([xs, ys]).T
    scale_factors = np.array([1 / max_distance_x, 1 / max_distance_y])

    def candidate_rule(x, y):
        point = np.array([x, y])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.mean(distances) < 1:
            candidate = True
        return candidate

    return candidate_rule


def plx_distance_rule(df, max_distance_gmag, plx_sigma, gmag_name, plx_name):
    gamgs = df[gmag_name].values
    plxs = df[plx_name].values
    plx_errors = df['parallax_error'].values

    members = np.array([gamgs, plxs]).T
    max_delta_squared = np.array([np.tile(np.array([max_distance_gmag]), len(plx_errors)),
                                  (plx_sigma * plx_errors)**2]).T

    def plx_candidate_rule(gmag, plx, plx_error):
        point = np.array([gmag, plx])
        deltas_squared = (members - point)**2

        candidate = False
        if np.any(np.all(deltas_squared < max_delta_squared + np.array([0., (plx_sigma * plx_error)**2]), axis=1)):
            candidate = True
        return candidate

    return plx_candidate_rule


def make_candidate_rule(member_df, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d):
    plx_candidate_rule = plx_distance_rule(member_df, gmag_max_d, plx_sigma, 'phot_g_mean_mag', 'parallax')
    pm_candidate_rule = pm_distance_rule(member_df, pm_max_d, pm_max_d, 'pmra', 'pmdec')
    isochrone_candidate_rule = isochrone_distance_rule(member_df, bp_rp_max_d, gmag_max_d, 'bp_rp', 'phot_g_mean_mag')

    def candidate_rule(row):
        candidate = False
        if plx_candidate_rule(row['phot_g_mean_mag'], row['parallax'], row['parallax_error']) and \
                pm_candidate_rule(row['pmra'], row['pmdec']) and \
                isochrone_candidate_rule(row['bp_rp'], row['phot_g_mean_mag']):
            candidate = True
        return candidate

    return candidate_rule


def divide_cone(cone_df,
                cluster_df,
                train_columns,
                probability_threshold,
                plx_sigma,
                gmag_max_d,
                pm_max_d,
                bp_rp_max_d,
                drop_db_candidates=False):
    n_sources = len(cone_df)
    print(f'Total number of sources in cone: {n_sources}')

    # select high probability members from the cone
    high_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >= 
                                              probability_threshold)]['Source'].astype(np.int64).values
    high_prob_member_df = cone_df[(cone_df['source_id'].isin(high_prob_member_source_ids))][train_columns +
                                                                                            ['parallax_error']].dropna()
    n_high_prob_member = len(high_prob_member_df)
    print(f'{n_high_prob_member} cluster sources were selected with a membership probability of >= '
          f'{probability_threshold}')
    
    # select low probability members from the cone
    low_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) <
                                             probability_threshold)]['Source'].astype(np.int64).values
    low_prob_member_df = cone_df[(cone_df['source_id'].isin(low_prob_member_source_ids))][train_columns].dropna()
    n_low_prob_members = len(low_prob_member_df)
    print(f'{n_low_prob_members} cluster sources had a too low probability to be used for the training set')
    
    # drop high probability members from the cone to obtain the field
    field_df = cone_df.drop(high_prob_member_df.index)[train_columns + ['parallax_error']].dropna()
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

    return high_prob_member_df, low_prob_member_df[train_columns], candidate_df[train_columns], non_member_df[train_columns]
