import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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

other_fields = ['source_id', 'ruwe', 'phot_g_mean_mag', 'bp_rp']

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


def normalize(dataframe, labels_not_normalized=None, all_data=None):
    if all_data is not None:
        data_norm = (dataframe - all_data.mean()) / all_data.std()
    else:
        data_norm = (dataframe - dataframe.mean()) / dataframe.std()
    if labels_not_normalized is not None:
        data_norm[labels_not_normalized] = dataframe[labels_not_normalized]
    return data_norm


def load_cone_data(data_dir, cone_file):
    practice_data_cone_file = os.path.join(data_dir, cone_file)

    # create cone dataframe
    cone_df = pd.read_csv(practice_data_cone_file)
    cone_df.columns = cone_df.columns.str.lower()
    return cone_df


def load_cluster_data(data_dir, cluster_file):
    practice_data_cluster_file = os.path.join(data_dir, cluster_file)

    # create cluster dataframe
    cluster_df = pd.read_csv(practice_data_cluster_file, sep='\t', header=61)
    cluster_df = cluster_df.iloc[2:]
    cluster_df = cluster_df.reset_index(drop=True)
    return cluster_df


def load_isochrone_data(data_dir, isochrone_file):
    practice_data_isochrone_file = os.path.join(data_dir, isochrone_file)

    # create isochrones dataframe
    isochrones_df = pd.read_csv(practice_data_isochrone_file, delim_whitespace=True, comment='#')
    return isochrones_df


def calculate_intersects(points, point0, point1):
    n_sources = len(points)

    xs = points[:, 0]
    ys = points[:, 1]

    if point0[0] > point1[0]:
        point0, point1 = point1, point0

    x0 = point0[0]
    y0 = point0[1]

    x1 = point1[0]
    y1 = point1[1]

    if x0 == x1:
        x_intersects = np.tile(x0, n_sources)
        y_intersects = ys
    elif y0 == y1:
        x_intersects = xs
        y_intersects = np.tile(y0, n_sources)
    else:
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        perp_a = - 1 / a
        perp_b = ys - perp_a * xs

        x_intersects = (perp_b - b) / (a - perp_a)
        y_intersects = a * x_intersects + b
    intersect_points = np.concatenate((x_intersects[:, None], y_intersects[:, None]), axis=1)

    return intersect_points


def point_line_section_delta(point, point0, point1):
    intersect_point = calculate_intersects(np.array([point]), point0, point1)[0]

    if point0[0] < intersect_point[0] < point1[0]:
        return intersect_point - point
    else:
        d0 = np.sqrt(np.sum(np.square(point0 - point)))
        d1 = np.sqrt(np.sum(np.square(point1 - point)))
        if d0 < d1:
            return point0 - point
        else:
            return point1 - point


def isochrone_distance_filter(isochrone, max_distance_bp_rp, max_distance_gmag):
    bp_rps = isochrone['bp_rp'].values
    gmags = isochrone['phot_g_mean_mag'].values
    n_isochrone_points = len(isochrone)

    isochrone_arr = np.array([bp_rps, gmags]).T
    scale_factors = np.array([1 / max_distance_bp_rp, 1 / max_distance_gmag])

    def candidate_filter(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])

        deltas = np.array([point_line_section_delta(point, isochrone_arr[i], isochrone_arr[i + 1])
                           for i in range(n_isochrone_points - 1)])
        scaled_deltas = deltas * scale_factors
        distances = np.sum(scaled_deltas ** 2, axis=1)

        candidate = False
        if np.any(distances < 1):
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


def plx_distance_filter(df, max_distance_gmag, plx_sigma):
    gamgs = df['phot_g_mean_mag'].values
    plxs = df['parallax'].values
    plx_errors = df['parallax_error'].values

    members = np.array([gamgs, plxs]).T
    max_delta_squared = np.array([np.tile(np.array([max_distance_gmag]), len(plx_errors)),
                                  (plx_sigma * plx_errors)**2]).T

    def plx_candidate_filter(row):
        gmag, plx, plx_error = row['phot_g_mean_mag'], row['parallax'], row['parallax_error']

        point = np.array([gmag, plx])
        deltas_squared = (members - point)**2

        candidate = False
        if np.any(np.all(deltas_squared < max_delta_squared + np.array([0., (plx_sigma * plx_error)**2]), axis=1)):
            candidate = True
        return candidate

    return plx_candidate_filter


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


def make_candidate_filter(filters):

    def candidate_filter(row):
        candidate = True
        for filter_rule in filters:
            passed_filter = filter_rule(row)
            if not passed_filter:
                candidate = False
                break
        return candidate

    return candidate_filter


def caluculate_distances(points, point0, point1, scale_factors):
    intersect_points = calculate_intersects(points, point0, point1)
    deltas = []

    for j in range(len(points)):
        point = points[j]
        if point0[0] < intersect_points[j][0] < point1[0]:
            deltas.append(intersect_points[j] - point)
        else:
            d0 = np.sqrt(np.sum(np.square(point0 - point)))
            d1 = np.sqrt(np.sum(np.square(point1 - point)))
            if d0 < d1:
                deltas.append(point0 - point)
            else:
                deltas.append(point1 - point)

    scaled_deltas = np.array(deltas) * scale_factors
    distances = np.sum(scaled_deltas ** 2, axis=1)
    return distances


def isochrone_distances(sources, isochrone, max_distance_bp_rp, max_distance_gmag):
    bp_rps = sources['bp_rp'].values
    gmags = sources['phot_g_mean_mag'].values
    n_sources = len(sources)
    n_isochrone_points = len(isochrone)

    isochrone_bp_rps = isochrone['bp_rp'].values
    isochrone_gmags = isochrone['phot_g_mean_mag'].values

    isochrone_arr = np.array([isochrone_bp_rps, isochrone_gmags]).T
    scale_factors = np.array([1 / max_distance_bp_rp, 1 / max_distance_gmag])

    distances = np.zeros(n_sources)
    cmd_sources = np.concatenate((bp_rps[:, None], gmags[:, None]), axis=1)

    for i in tqdm(range(n_isochrone_points - 1), desc="Calculating isochrone distances..."):
        isochrone_point0 = isochrone_arr[i]
        isochrone_point1 = isochrone_arr[i + 1]

        new_distances = caluculate_distances(cmd_sources, isochrone_point0, isochrone_point1, scale_factors)

        if i == 0:
            distances = new_distances
        else:
            distances = np.min(np.stack((distances, new_distances)), axis=0)

    return distances


def make_hp_member_df(cone_df, cluster_df, probability_threshold):
    high_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >=
                                              probability_threshold)]['Source'].astype(np.int64).values
    high_prob_member_df = cone_df[(cone_df['source_id'].isin(high_prob_member_source_ids))]
    return high_prob_member_df[fields['all']].dropna()


def make_lp_member_df(cone_df, cluster_df, probability_threshold):
    low_prob_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) <
                                             probability_threshold)]['Source'].astype(np.int64).values
    low_prob_member_df = cone_df[(cone_df['source_id'].isin(low_prob_member_source_ids))]
    return low_prob_member_df[fields['all']].dropna()


def make_field_df(cone_df, high_prob_member_df):
    field_df = cone_df.drop(high_prob_member_df.index)
    return field_df[fields['all']].dropna()


def make_noise_candidate_df(cone_df, members, isochrone, candidate_filter_kwargs):
    field = make_field_df(cone_df, members)

    filters = [pm_distance_filter(members,
                                  candidate_filter_kwargs['pm_max_d'],
                                  candidate_filter_kwargs['pm_max_d']),
               plx_distance_filter(members,
                                   candidate_filter_kwargs['gmag_max_d'],
                                   candidate_filter_kwargs['plx_sigma']),
               isochrone_distance_filter(isochrone,
                                         candidate_filter_kwargs['bp_rp_max_d'],
                                         candidate_filter_kwargs['gmag_max_d'])]
    candidate_filter = make_candidate_filter(filters)

    candidates = field[field.apply(candidate_filter, axis=1)]
    noise = field[field.apply(lambda row: not candidate_filter(row), axis=1)]
    return noise, candidates


def isochrone_correction(isochrones, mean_plx, extinction_v):
    # correct the isochrones for distance and extinction
    mean_distance = 1000 / mean_plx
    isochrone_distance = mean_distance

    # print('\nMean distance of isochrone: ', isochrone_distance, ' pc')

    extinction_g = 0.83627 * extinction_v
    extinction_bp = 1.08337 * extinction_v
    extinction_rp = 0.6343 * extinction_v

    distance_modulus = 5 * np.log10(0.80 * isochrone_distance) - 5

    gmag_correction = distance_modulus + extinction_g
    bpmag_correction = distance_modulus + extinction_bp
    rpmag_correction = distance_modulus + extinction_rp

    for isochrone in isochrones:
        isochrone['Gmag'] += gmag_correction
        isochrone.rename(columns={'Gmag': 'phot_g_mean_mag'}, inplace=True)
        isochrone['G_BPmag'] += bpmag_correction
        isochrone['G_RPmag'] += rpmag_correction
        isochrone['bp_rp'] = isochrone['G_BPmag'] - isochrone['G_RPmag']

    return isochrones


def make_isochrone(data_dir, isochrone_file, mean_plx, extinction_v, cutoff):
    isochrones_df = load_isochrone_data(data_dir, isochrone_file)
    isochrones = []
    for age in list(set(isochrones_df['logAge'].values)):
        isochrones.append(isochrones_df[(isochrones_df['logAge'] == age)].iloc[:cutoff])
    isochrone = isochrone_correction(isochrones, mean_plx, extinction_v=extinction_v)[0]
    return isochrone


def parse_data(data_dir, cone_file, cluster_file, isochrone_file, probability_threshold, candidate_filter_kwargs):
    cone_df = load_cone_data(data_dir, cone_file)
    cluster_df = load_cluster_data(data_dir, cluster_file)

    hp_members = make_hp_member_df(cone_df, cluster_df, probability_threshold)
    lp_members = make_lp_member_df(cone_df, cluster_df, probability_threshold)

    mean_plx = np.mean(hp_members['parallax'].values)
    isochrone = make_isochrone(data_dir, isochrone_file, mean_plx, extinction_v=0.13, cutoff=135)
    noise, candidates = make_noise_candidate_df(cone_df, hp_members, isochrone, candidate_filter_kwargs)

    for sources in [lp_members, hp_members, noise, candidates]:
        sources['isochrone_d'] = isochrone_distances(sources,
                                                     isochrone,
                                                     candidate_filter_kwargs['bp_rp_max_d'],
                                                     candidate_filter_kwargs['gmag_max_d'])

    return lp_members, hp_members, noise, candidates, isochrone
