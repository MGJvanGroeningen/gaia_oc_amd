import numpy as np


def make_plx_candidate_rule(member_df, sigma_factor):
    max_gmag = member_df['phot_g_mean_mag'].max()
    min_gmag = member_df['phot_g_mean_mag'].min()
    median_gmag = member_df['phot_g_mean_mag'].median()

    low_gmag_member_df = member_df[(member_df['phot_g_mean_mag'] <= median_gmag)]
    high_gmag_member_df = member_df[(member_df['phot_g_mean_mag'] > median_gmag)]

    low_gmag_mean_plx = low_gmag_member_df['parallax'].mean()
    high_gmag_mean_plx = high_gmag_member_df['parallax'].mean()

    low_gmag_std_plx = low_gmag_member_df['parallax'].std()
    high_gmag_std_plx = high_gmag_member_df['parallax'].std()

    x1 = (median_gmag - min_gmag) / 2
    x2 = (max_gmag - median_gmag) / 2

    y1_top = low_gmag_mean_plx + sigma_factor * low_gmag_std_plx
    y2_top = high_gmag_mean_plx + sigma_factor * high_gmag_std_plx

    y1_bottom = low_gmag_mean_plx - sigma_factor * low_gmag_std_plx
    y2_bottom = high_gmag_mean_plx - sigma_factor * high_gmag_std_plx

    a_top = (y2_top - y1_top) / (x2 - x1)
    b_top = y1_top - a_top * x1

    a_bottom = (y2_bottom - y1_bottom) / (x2 - x1)
    b_bottom = y1_bottom - a_bottom * x1

    def candidate_rule(gmag, plx):
        candidate = False
        if min_gmag - 1 < gmag < max_gmag + 1 and \
                a_top * plx + b_top > plx > a_bottom * plx + b_bottom:
            candidate = True
        return candidate

    return candidate_rule


def make_pm_candidate_rule(member_df, sigma_factor):
    mean_pmra = member_df['pmra'].mean()
    mean_pmdec = member_df['pmdec'].mean()

    std_pmra = member_df['pmra'].std()
    std_pmdec = member_df['pmdec'].std()

    min_pmra = mean_pmra - sigma_factor * std_pmra
    min_pmdec = mean_pmdec - sigma_factor * std_pmdec

    max_pmra = mean_pmra + sigma_factor * std_pmra
    max_pmdec = mean_pmdec + sigma_factor * std_pmdec

    def candidate_rule(pmra, pmdec):
        candidate = False
        if min_pmra < pmra < max_pmra and min_pmdec < pmdec < max_pmdec:
            candidate = True
        return candidate

    return candidate_rule


def make_isochrone_candidate_rule(member_df, sigma_factor):
    gmags = member_df['phot_g_mean_mag'].values
    bp_rps = member_df['bp_rp'].values

    members = np.array([bp_rps, gmags]).T

    def candidate_rule(bp_rp, gmag):
        point = np.array([bp_rp, gmag])
        deltas = members - point
        distances_squared = np.einsum('ij,ij->i', deltas, deltas)
        closest_distance = np.sqrt(np.min(distances_squared))
        # prob = np.sqrt(2 * np.pi) * np.mean(np.exp(- 0.5 * distances_squared))

        candidate = False
        if closest_distance < sigma_factor:
            candidate = True
        return candidate

    return candidate_rule


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


def make_candidate_rule(member_df, plx_sigma_factor, pm_sigma_factor, isochrone_sigma_factor):
    # plx_candidate_rule = make_plx_candidate_rule(member_df, plx_sigma_factor)
    # pm_candidate_rule = make_pm_candidate_rule(member_df, pm_sigma_factor)
    # isochrone_candidate_rule = make_isochrone_candidate_rule(member_df, isochrone_sigma_factor)

    plx_candidate_rule = distance_rule(member_df, plx_sigma_factor, 'phot_g_mean_mag', 'parallax')
    pm_candidate_rule = distance_rule(member_df, pm_sigma_factor, 'pmra', 'pmdec')
    isochrone_candidate_rule = distance_rule(member_df, isochrone_sigma_factor, 'bp_rp', 'phot_g_mean_mag')

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
                plx_sigma_factor,
                pm_sigma_factor,
                isochrone_sigma_factor,
                drop_database_candidates=False):
    print(f'Total number of sources in cone: {len(cone_df)}')

    # make a new column in the cone dataframe that indicates whether a source belongs to the cluster
    member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >=
                                    probability_threshold)]['Source'].astype(np.int64).values
    member_df = cone_df[(cone_df['source_id'].isin(member_source_ids))][train_columns].dropna()

    print(f'{len(member_df)} cluster sources were selected with a membership probability of >= {probability_threshold}')

    field_df = cone_df.drop(member_df.index)[train_columns].dropna()

    database_candidate_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) <
                                                probability_threshold)]['Source'].astype(np.int64).values
    database_candidate_df = cone_df[(cone_df['source_id'].isin(database_candidate_source_ids))][train_columns].dropna()

    print(f'{len(database_candidate_df)} cluster sources had a too low probability to be used for the training set')

    if drop_database_candidates:
        field_df = field_df.drop(database_candidate_df.index)

    candidate_rule = make_candidate_rule(member_df,
                                         plx_sigma_factor,
                                         pm_sigma_factor,
                                         isochrone_sigma_factor)

    field_df['candidate'] = field_df.apply(candidate_rule, axis=1)

    candidate_df = field_df[(field_df['candidate'])]

    print(f'{len(candidate_df)} sources were selected as candidates')

    candidate_df = candidate_df.drop('candidate', axis=1)
    field_df = field_df.drop('candidate', axis=1)

    if len(candidate_df) > 0:
        non_member_df = field_df.append(candidate_df).drop_duplicates(keep=False)
    else:
        non_member_df = field_df

    print(f'{len(non_member_df)} sources were selected as non members')

    return member_df, database_candidate_df, candidate_df, non_member_df
