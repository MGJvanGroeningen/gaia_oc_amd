import numpy as np


def distance_rule(df, max_distance, x_name, y_name):
    xs = df[x_name].values
    ys = df[y_name].values

    members = np.array([xs, ys]).T

    def candidate_rule(x, y):
        point = np.array([x, y])
        deltas = members - point
        distances_squared = np.einsum('ij,ij->i', deltas, deltas)
        closest_distance = np.sqrt(np.min(distances_squared))

        candidate = False
        if closest_distance < max_distance:
            candidate = True
        return candidate

    return candidate_rule


def make_candidate_rule(member_df, plx_max_d, pm_max_d, isochrone_max_d):
    plx_candidate_rule = distance_rule(member_df, plx_max_d, 'phot_g_mean_mag', 'parallax')
    pm_candidate_rule = distance_rule(member_df, pm_max_d, 'pmra', 'pmdec')
    isochrone_candidate_rule = distance_rule(member_df, isochrone_max_d, 'bp_rp', 'phot_g_mean_mag')

    def candidate_rule(row):
        candidate = False
        if plx_candidate_rule(row['phot_g_mean_mag'], row['parallax']) and \
                pm_candidate_rule(row['pmra'], row['pmdec']) and \
                isochrone_candidate_rule(row['bp_rp'], row['phot_g_mean_mag']):
            candidate = True
        return candidate

    return candidate_rule


def divide_cone(cone_df,
                cluster_df,
                train_columns,
                probability_threshold,
                plx_max_d,
                pm_max_d,
                isochrone_max_d,
                drop_db_candidates=False):
    n_sources = len(cone_df)
    print(f'Total number of sources in cone: {n_sources}')

    # select high probability members from the cone
    high_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >= 
                                              probability_threshold)]['Source'].astype(np.int64).values
    high_prob_member_df = cone_df[(cone_df['source_id'].isin(high_prob_member_source_ids))][train_columns].dropna()
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
    field_df = cone_df.drop(high_prob_member_df.index)[train_columns].dropna()
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
    candidate_rule = make_candidate_rule(high_prob_member_df, plx_max_d, pm_max_d, isochrone_max_d)
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
