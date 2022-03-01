import numpy as np
import pandas as pd
import time
import os
from astropy.io.votable import parse

astrometric_value_fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
astrometric_error_fields = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']
astrometric_corr_fields = ['ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                           'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
                           'parallax_pmra_corr', 'parallax_pmdec_corr',
                           'pmra_pmdec_corr']
astrometric_fields = astrometric_value_fields + astrometric_error_fields + astrometric_corr_fields

photometric_value_fields = ['phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux']
photometric_error_fields = ['phot_g_mean_flux_error', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux_error']
photometric_fields = photometric_value_fields + photometric_error_fields

other_fields = ['source_id', 'ruwe', 'phot_g_mean_mag', 'bp_rp', 'PMemb']

all_fields = astrometric_fields + photometric_fields + other_fields

plot_fields = ['ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']

fields = {'astrometric': astrometric_value_fields,
          'astrometric_error': astrometric_error_fields,
          'astrometric_corr': astrometric_corr_fields,
          'photometric': photometric_value_fields,
          'photometric_error': photometric_error_fields,
          'other': other_fields,
          'all': all_fields,
          'plot': plot_fields}


def load_cone_data(cone_path):
    # create cone dataframe
    # cone_df = pd.read_csv(cone_path, chunksize=chunksize)

    print('Loading cone data...', end=' ')

    cone_df = parse(cone_path)
    cone_df = cone_df.get_first_table().to_table(use_names_over_ids=True)
    cone_df = cone_df.to_pandas()
    print('Done')
    return cone_df


def load_cluster_data(cluster_path):
    # create cluster dataframe
    cluster_df = pd.read_csv(cluster_path)
    # cluster_df = pd.read_csv(cluster_path, sep='\t', header=61)
    # cluster_df = cluster_df.iloc[2:]
    # cluster_df = cluster_df.reset_index(drop=True)
    return cluster_df


def load_isochrone_data(isochrone_path):
    # create isochrones dataframe
    isochrones_df = pd.read_csv(isochrone_path, delim_whitespace=True, comment='#')
    return isochrones_df


def get_cluster_parameters(cluster_parameters_path, cluster_name):
    cluster_params_df = pd.read_csv(cluster_parameters_path, sep='\t', header=50, skipinitialspace=True)
    cluster_params_df = cluster_params_df.astype(str).iloc[2:].reset_index(drop=True).applymap(lambda x: x.strip())
    cluster_params_df = cluster_params_df.drop(cluster_params_df[cluster_params_df['AgeNN'].str.strip() == 'nan'].index)

    float_columns = {column: np.float32 for column in ['RA_ICRS', 'DE_ICRS', 'pmRA*', 'pmDE', 'plx', 'AgeNN', 'AVNN',
                                                       'DMNN', 'DistPc']}

    cluster_params_df = cluster_params_df.astype(float_columns)
    cluster_params_df = cluster_params_df.astype({'nbstars07': np.int32})

    params = cluster_params_df[cluster_params_df['Cluster'] == cluster_name]

    cluster_parameters = {'ra': params['RA_ICRS'].values[0],
                          'dec': params['DE_ICRS'].values[0],
                          'pmra': params['pmRA*'].values[0],
                          'pmdec': params['pmDE'].values[0],
                          'age': params['AgeNN'].values[0],
                          'a_v': params['AVNN'].values[0],
                          'plx': params['plx'].values[0],
                          'dm': params['DMNN'].values[0],
                          'dist': params['DistPc'].values[0]}

    return cluster_parameters


def ellipse_isochrone_contacts(point, isochrone_pairs, x_scale, y_scale):
    x = point[0]
    y = point[1]

    x0s = isochrone_pairs[:, 0, 0] - x
    y0s = isochrone_pairs[:, 0, 1] - y
    x1s = isochrone_pairs[:, 1, 0] - x
    y1s = isochrone_pairs[:, 1, 1] - y

    delta_y = y1s - y0s
    delta_x = np.where(x1s - x0s == 0, 1, x1s - x0s)

    m = delta_y / delta_x
    c = y0s - m * x0s

    # determine the parameters of the source-centered ellipse that just hits the isochrone line section
    k = y_scale / x_scale
    a2 = c ** 2 / (m ** 2 + k ** 2)
    b = k * np.sqrt(a2)

    # determine the coordinates of the contact point
    x_contacts = - a2 * m * c / (a2 * m ** 2 + b ** 2)
    y_contacts = m * x_contacts + c

    contact_points = np.concatenate(((x_contacts + x)[:, None], (y_contacts + y)[:, None]), axis=1)

    return contact_points


def norm(arr, axis=0):
    return np.sqrt(np.sum(np.square(arr), axis=axis))


def source_isochrone_section_delta(point, isochrone_pairs, x_scale, y_scale):
    scale_factors = np.array([1 / x_scale, 1 / y_scale])

    contact_points = ellipse_isochrone_contacts(point, isochrone_pairs, x_scale, y_scale)
    contact_point_delta = (contact_points - point) * scale_factors

    delta_1 = (isochrone_pairs[:, 0] - point) * scale_factors
    delta_2 = (isochrone_pairs[:, 1] - point) * scale_factors

    closest_isochrone_point_delta = np.where(np.tile(norm(delta_1, axis=1) < norm(delta_2, axis=1), (2, 1)).T,
                                             delta_1, delta_2)

    distances = np.where(np.tile((np.min(isochrone_pairs[:, :, 0], axis=1) < contact_points[:, 0]) &
                         (np.max(isochrone_pairs[:, :, 0], axis=1) > contact_points[:, 0]), (2, 1)).T,
                         contact_point_delta,
                         closest_isochrone_point_delta)

    shortest_isochrone_delta = distances[np.argmin(norm(distances, axis=1))]
    return shortest_isochrone_delta


def isochrone_distance_filter(isochrone_pairs, max_distance_bp_rp, max_distance_gmag):

    def candidate_filter(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])

        delta = source_isochrone_section_delta(point, isochrone_pairs, max_distance_bp_rp, max_distance_gmag)

        candidate = False
        if norm(delta) < 1:
            candidate = True
        return candidate

    return candidate_filter


def pm_distance_filter(df, max_distance_pmra, max_distance_pmdec):
    pmras = df['pmra'].values
    pmdecs = df['pmdec'].values

    members = np.array([pmras, pmdecs]).T
    scale_factors = np.array([1 / max_distance_pmra, 1 / max_distance_pmdec])

    def candidate_filter(row):
        pmra, pmdec = row['pmra'], row['pmdec']
        point = np.array([pmra, pmdec])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.mean(distances) < 1:
            candidate = True
        return candidate

    return candidate_filter


def plx_distance_filter(df):
    cluster_mean_plx = most_likely_value(df['parallax'], df['parallax_error'])
    cluster_std_plx_2 = (cluster_mean_plx * (1 - 1 / (1 + 0.03 * cluster_mean_plx))) ** 2
    # min_noise_distance = 0.05

    def plx_candidate_filter(row):
        plx = row['parallax']
        plx_e = np.sqrt(row['parallax_error']**2 + cluster_std_plx_2)

        plx_d = (cluster_mean_plx - plx)**2
        max_plx_d = (3 * plx_e)**2
        # max_plx_d = (2.0 / (-(gmag - 21.5)) ** 2.3 + min_noise_distance)**2

        candidate = False
        if plx_d < max_plx_d:
            candidate = True
        return candidate

    return plx_candidate_filter


# def plx_distance_filter_2(df, max_distance_gmag, plx_sigma):
#     gmags = df['phot_g_mean_mag'].values
#     plxs = df['parallax'].values
#     plx_errors = df['parallax_error'].values
#
#     members = np.array([gmags, plxs]).T
#     max_delta_squared = np.array([np.tile(np.array([max_distance_gmag]), len(plx_errors)),
#                                   (plx_sigma * plx_errors)**2]).T
#
#     def plx_candidate_filter(row):
#         gmag, plx, plx_error = row['phot_g_mean_mag'], row['parallax'], row['parallax_error']
#
#         point = np.array([gmag, plx])
#         deltas_squared = (members - point)**2
#
#         candidate = False
#         if np.any(np.all(deltas_squared < max_delta_squared + np.array([0., (plx_sigma * plx_error)**2]), axis=1)):
#             candidate = True
#         return candidate
#
#     return plx_candidate_filter


def cmd_distance_filter(df, max_distance_bp_rp, max_distance_gmag):
    bp_rps = df['bp_rp'].values
    gmags = df['phot_g_mean_mag'].values

    members = np.array([bp_rps, gmags]).T
    scale_factors = np.array([1 / max_distance_bp_rp, 1 / max_distance_gmag])

    def candidate_filter(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])
        deltas = members - point
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas**2, axis=1)

        candidate = False
        if np.any(distances < 1):
            candidate = True
        return candidate

    return candidate_filter


def make_isochrone_pairs(isochrone):
    bp_rp_pairs = np.stack((isochrone['bp_rp'].values[:-1], isochrone['bp_rp'].values[1:]), axis=-1)
    gmag_pairs = np.stack((isochrone['phot_g_mean_mag'].values[:-1], isochrone['phot_g_mean_mag'].values[1:]), axis=-1)
    isochrone_pairs = np.stack((bp_rp_pairs, gmag_pairs), axis=-1)
    return isochrone_pairs


def make_candidate_filter(members, isochrone, candidate_filter_kwargs):
    isochrone_pairs = make_isochrone_pairs(isochrone)

    filters = [pm_distance_filter(members,
                                  candidate_filter_kwargs['pm_max_d'],
                                  candidate_filter_kwargs['pm_max_d']),
               plx_distance_filter(members),
               isochrone_distance_filter(isochrone_pairs,
                                         candidate_filter_kwargs['bp_rp_max_d'],
                                         candidate_filter_kwargs['gmag_max_d'])]

    def candidate_filter(row):
        candidate = True
        for filter_rule in filters:
            passed_filter = filter_rule(row)
            if not passed_filter:
                candidate = False
                break
        return candidate

    return candidate_filter


def make_bias_filter(isochrone_pairs, bias):

    def candidate_filter(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']

        bias_scale = 1

        x = bp_rp
        y = gmag

        left_of_isochrone = []

        for pair in isochrone_pairs:
            if np.min(pair[:, 1]) < y < np.max(pair[:, 1]):
                delta_y = pair[1, 1] - pair[0, 1]
                delta_x = np.where(pair[1, 0] - pair[0, 0] == 0, 1e-8, pair[1, 0] - pair[0, 0])

                m = delta_y / delta_x
                c = pair[0, 1] - m * pair[0, 0]
                left_of_isochrone.append(x < (y - c) / m)

        if len(left_of_isochrone) != 0 and all(left_of_isochrone):
            bias_scale = (y / 21)**2.3 * bias

        candidate = True
        if bias_scale * row['isochrone_d'] > 1:
            candidate = False
        return candidate

    return candidate_filter


def make_member_df(cone_df, cluster_df, probability_threshold):
    cone_df.columns = cone_df.columns.str.lower()
    cluster_df['Source'] = cluster_df['Source'].astype(np.int64)
    cluster_df_2 = cluster_df[['Source', 'PMemb']].copy()

    members = pd.merge(cluster_df_2, cone_df, left_on='Source', right_on='source_id', how='inner',
                       copy=False)[fields['all']].dropna()
    hp_member_subset = members[(members['PMemb'] >= probability_threshold)].copy()
    lp_member_subset = members[(members['PMemb'] < probability_threshold)].copy()
    return hp_member_subset, lp_member_subset


def make_field_df(cone_df, member_df):
    field_df = cone_df[~cone_df['source_id'].isin(member_df['source_id'])].copy()
    field_df['PMemb'] = 0
    return field_df[fields['all']].dropna()


def make_noise_candidate_df(field, candidate_filter):
    candidate_indices = field.apply(candidate_filter, axis=1)
    candidates = field[candidate_indices].copy()
    noise = field[~candidate_indices].copy()
    return noise, candidates


def most_likely_value(values, errors):
    return np.sum(values / errors ** 2) / np.sum(1 / errors ** 2)


def danielski_g(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def danielski_bp(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def danielski_rp(a0, colour):
    c1, c2, c3, c4, c5, c6, c7 = 0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006
    k = c1 + c2 * colour + c3 * colour ** 2 + c4 * colour ** 3 + c5 * a0 + c6 * a0 ** 2 + c7 * colour * a0
    return k * a0


def isochrone_correction(isochrones, distance_modulus):
    # extinction_g = 0.83627 * extinction_v
    # extinction_bp = 1.08337 * extinction_v
    # extinction_rp = 0.6343 * extinction_v

    gmag_correction = distance_modulus  # + extinction_g
    bpmag_correction = distance_modulus  # + extinction_bp
    rpmag_correction = distance_modulus  # + extinction_rp

    for isochrone in isochrones:
        isochrone['Gmag'] += gmag_correction
        isochrone.rename(columns={'Gmag': 'phot_g_mean_mag'}, inplace=True)
        isochrone['G_BPmag'] += bpmag_correction
        isochrone['G_RPmag'] += rpmag_correction
        isochrone['bp_rp'] = isochrone['G_BPmag'] - isochrone['G_RPmag']

    return isochrones


def make_isochrone(isochrone_path, age, z, dm):
    isochrones_df = load_isochrone_data(isochrone_path)
    ages = np.array(list(set(isochrones_df['logAge'].values)))
    zs = np.array(list(set(isochrones_df['Zini'].values)))

    closest_age = ages[np.argmin(np.abs(ages - age))]
    closest_z = zs[np.argmin(np.abs(zs - z))]

    isochrone = isochrones_df[(isochrones_df['logAge'] == closest_age) &
                              (isochrones_df['Zini'] == closest_z)].copy()

    isochrone = isochrone_correction([isochrone], dm)[0].iloc[5:]
    previous_bp_rp = np.inf
    end = -1
    for idx, bp_rp in enumerate(isochrone['bp_rp']):
        if bp_rp > 2.5 and bp_rp > previous_bp_rp:
            end = idx
            break
        previous_bp_rp = bp_rp

    isochrone = isochrone.iloc[:end]

    return isochrone


def isochrone_delta_f(isochrone, candidate_filter_kwargs):

    isochrone_pairs = make_isochrone_pairs(isochrone)

    bp_rp_max_d = candidate_filter_kwargs['bp_rp_max_d']
    gmag_max_d = candidate_filter_kwargs['gmag_max_d']

    def isochrone_delta(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])

        delta = source_isochrone_section_delta(point, isochrone_pairs, bp_rp_max_d, gmag_max_d)
        return delta

    return isochrone_delta


def mean_pm_f(members):
    mean_pmra = most_likely_value(members['pmra'], members['pmra_error'])
    mean_pmdec = most_likely_value(members['pmdec'], members['pmdec_error'])

    def pm_distance(row):
        pmra, pmdec = row['pmra'], row['pmdec']
        pm_d = np.sqrt((pmra - mean_pmra) ** 2 + (pmdec - mean_pmdec) ** 2)
        return pm_d

    return pm_distance


def mean_plx_f(members):
    cluster_mean_plx = most_likely_value(members['parallax'], members['parallax_error'])
    cluster_std_plx_2 = (cluster_mean_plx * (1 - 1 / (1 + 0.01 * cluster_mean_plx)))**2

    def plx_distance(row):
        plx = row['parallax']
        plx_e = np.sqrt(row['parallax_error']**2 + cluster_std_plx_2)
        plx_d = np.exp(- ((plx - cluster_mean_plx) / plx_e)**2 / 2) / (np.sqrt(2 * np.pi) * plx_e)
        return plx_d

    return plx_distance


def ra_dec_to_xy(ra, dec, ra_c, dec_c, d):
    x = d * np.sin(ra - ra_c) * np.cos(dec)
    y = d * (np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c))
    return x, y


def projected_coordinates(members, cluster_kwargs):
    rad_per_deg = np.pi / 180

    ra = members['ra'] * rad_per_deg
    dec = members['dec'] * rad_per_deg

    if cluster_kwargs is not None:
        d = cluster_kwargs['dist']
        ra_c = cluster_kwargs['ra'] * rad_per_deg
        dec_c = cluster_kwargs['dec'] * rad_per_deg
    else:
        d = 1000 / (most_likely_value(members['parallax'], members['parallax_error']) + 0.029)
        ra_c = most_likely_value(members['ra'], members['ra_error']) * rad_per_deg
        dec_c = most_likely_value(members['dec'], members['dec_error']) * rad_per_deg

    x, y = ra_dec_to_xy(ra, dec, ra_c, dec_c, d)
    return x.values, y.values


def crop_member_candidates(member_candidates, r_t, prob_threshold, cluster_kwargs):
    x, y = projected_coordinates(member_candidates, cluster_kwargs)
    member_candidates['r'] = np.sqrt(x ** 2 + y ** 2)
    cropped_member_candidates = member_candidates[(member_candidates['r'] <= r_t) &
                                                  (member_candidates['PMemb'] >= prob_threshold)].copy()
    return cropped_member_candidates


def mean_ra_dec_f(cluster_kwargs):
    rad_per_deg = np.pi / 180

    d = cluster_kwargs['dist']
    ra_c = cluster_kwargs['ra'] * rad_per_deg
    dec_c = cluster_kwargs['dec'] * rad_per_deg

    def plx_distance(row):
        ra = row['ra'] * rad_per_deg
        dec = row['dec'] * rad_per_deg

        x, y = ra_dec_to_xy(ra, dec, ra_c, dec_c, d)

        ra_dec_d = np.sqrt(x**2 + y**2)
        return ra_dec_d

    return plx_distance


def extinction_correction(sources, cluster_kwargs):
    extinction_g = danielski_g(cluster_kwargs['a_v'], sources['bp_rp'])
    extinction_bp = danielski_bp(cluster_kwargs['a_v'], sources['bp_rp'])
    extinction_rp = danielski_rp(cluster_kwargs['a_v'], sources['bp_rp'])

    sources['phot_g_mean_mag'] -= extinction_g
    sources['bp_rp'] -= extinction_bp - extinction_rp
    return sources


def parse_members(cone_df, members_df, probability_threshold, cluster_kwargs):
    print('Cluster sources:', len(members_df))
    print('Cone sources', len(cone_df))

    hp_members, lp_members = make_member_df(cone_df, members_df, probability_threshold)

    for sources in [hp_members, lp_members]:
        extinction_correction(sources, cluster_kwargs)

    print(f'High probability members (>{probability_threshold}):', len(hp_members))
    print('Total members:', len(hp_members) + len(lp_members))
    print('Missing cluster sources:', len(members_df) - (len(hp_members) + len(lp_members)))

    return hp_members, lp_members


def parse_candidates(cone_df, hp_members, isochrone, cluster_kwargs, candidate_filter_kwargs, max_noise):
    candidate_filter = make_candidate_filter(hp_members, isochrone, candidate_filter_kwargs)
    field = make_field_df(cone_df, hp_members)
    extinction_correction(field, cluster_kwargs)
    noise, candidates = make_noise_candidate_df(field, candidate_filter)

    if len(noise) > max_noise:
        noise = noise.sample(max_noise)

    candidates['PMemb'] = 0
    noise['PMemb'] = 0

    print('Candidates:', len(candidates))
    print('Noise:', len(noise))

    return candidates, noise


def add_train_fields(parsed_sources, isochrone, candidate_filter_kwargs, cluster_kwargs):
    # Add a number of auxiliary fields (isochrone delta, parallax distance, proper motion distance)
    isochrone_delta = isochrone_delta_f(isochrone, candidate_filter_kwargs)
    pm_distance = mean_pm_f(parsed_sources['hp_members'])
    plx_distance = mean_plx_f(parsed_sources['hp_members'])
    ra_dec_distance = mean_ra_dec_f(cluster_kwargs)

    print('Adding training fields... ', end='')
    t0 = time.time()
    for subset in parsed_sources:
        if len(parsed_sources[subset]) > 0:
            parsed_sources[subset][['bp_rp_d', 'gmag_d']] = parsed_sources[subset].apply(isochrone_delta, axis=1,
                                                                                         result_type='expand')
            parsed_sources[subset]['pm_d'] = parsed_sources[subset].apply(pm_distance, axis=1)
            parsed_sources[subset]['plx_d'] = parsed_sources[subset].apply(plx_distance, axis=1)
            parsed_sources[subset]['ra_dec_d'] = parsed_sources[subset].apply(ra_dec_distance, axis=1)
    print(f'Done in {np.round(time.time() - t0, 2)} sec')

    return parsed_sources


def parse_data(cone_path, members_path, compare_members_path, isochrone, probability_threshold, candidate_filter_kwargs,
               cluster_kwargs, max_noise=10000):
    cone_df = load_cone_data(cone_path)
    members_df = load_cluster_data(members_path)

    hp_members, lp_members = parse_members(cone_df, members_df, probability_threshold, cluster_kwargs)

    if os.path.exists(compare_members_path):
        compare_df = load_cluster_data(compare_members_path)
        compare_members, _ = parse_members(cone_df, compare_df, probability_threshold, cluster_kwargs)
    else:
        compare_members = pd.DataFrame()

    candidates, noise = parse_candidates(cone_df, hp_members, isochrone, cluster_kwargs, candidate_filter_kwargs,
                                         max_noise)

    # Combine subsets in a dictionary
    parsed_sources = {'lp_members': lp_members,
                      'hp_members': hp_members,
                      'compare_members': compare_members,
                      'noise': noise,
                      'candidates': candidates}

    return parsed_sources
