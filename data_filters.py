import numpy as np
import pandas as pd
import time
import os
from astropy.io.votable import parse
from utils import members_path_to_ref
from sets import Sources
from diagnostics import projected_coordinates

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

photometric_mag_value_fields = ['phot_g_mean_mag', 'bp_rp']
photometric_mag_error_fields = ['phot_g_mean_mag_error', 'bp_rp_error']
photometric_mag_fields = photometric_mag_value_fields + photometric_mag_error_fields

query_fields = astrometric_fields + photometric_fields + ['source_id', 'ruwe', 'phot_g_mean_mag', 'bp_rp', 'l', 'b']
all_fields = query_fields + ['phot_g_mean_mag_error', 'bp_rp_error', 'PMemb']

plot_fields = ['l', 'b', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', 'bp_rp']

fields = {'astrometric': astrometric_value_fields,
          'astrometric_error': astrometric_error_fields,
          'astrometric_corr': astrometric_corr_fields,
          'photometric': photometric_mag_value_fields,
          'photometric_error': photometric_mag_error_fields,
          'query': query_fields,
          'all': all_fields,
          'plot': plot_fields}


def get_cluster_parameters(cluster_name, cluster_parameters_path):
    cluster_params_df = pd.read_csv(cluster_parameters_path, sep='\t', header=60, skipinitialspace=True)
    cluster_params_df = cluster_params_df.astype(str).iloc[2:].reset_index(drop=True).applymap(lambda x: x.strip())
    cluster_params_df = cluster_params_df.drop(cluster_params_df[cluster_params_df['AgeNN'].str.strip() == 'nan'].index)

    if cluster_name not in cluster_params_df['Cluster'].values:
        return None
    else:
        float_columns = {column: np.float32 for column in ['RA_ICRS', 'DE_ICRS', 'r50', 'pmRA*', 'pmDE', 'plx', 'e_pmRA*',
                                                           'e_pmDE', 'e_plx', 'AgeNN', 'AVNN', 'DMNN', 'DistPc']}

        cluster_params_df = cluster_params_df.astype(float_columns)
        cluster_params_df = cluster_params_df.astype({'nbstars07': np.int32})

        params = cluster_params_df[cluster_params_df['Cluster'] == cluster_name]
        return params


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

    # determine the coordinates of the contact point
    x_contacts = - m * c / (m ** 2 + k ** 2)
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

    deltas = np.where(np.tile((np.min(isochrone_pairs[:, :, 0], axis=1) < contact_points[:, 0]) &
                              (np.max(isochrone_pairs[:, :, 0], axis=1) > contact_points[:, 0]), (2, 1)).T,
                      contact_point_delta,
                      closest_isochrone_point_delta)

    shortest_isochrone_delta = deltas[np.argmin(norm(deltas, axis=1))]
    return shortest_isochrone_delta


def isochrone_delta(isochrone, candidate_filter_kwargs):
    bp_rp_pairs = np.stack((isochrone['bp_rp'].values[:-1], isochrone['bp_rp'].values[1:]), axis=-1)
    gmag_pairs = np.stack((isochrone['phot_g_mean_mag'].values[:-1], isochrone['phot_g_mean_mag'].values[1:]), axis=-1)

    isochrone_pairs = np.stack((bp_rp_pairs, gmag_pairs), axis=-1)
    bp_rp_max_d = candidate_filter_kwargs['bp_rp_d_max']
    gmag_max_d = candidate_filter_kwargs['gmag_d_max']

    def isochrone_del(row):
        bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
        point = np.array([bp_rp, gmag])

        delta = source_isochrone_section_delta(point, isochrone_pairs, bp_rp_max_d, gmag_max_d)
        return delta

    return isochrone_del


def pm_distance(cluster, candidate_filter=False):
    def pm_dist(row):
        pmra = row['pmra']
        pmdec = row['pmdec']
        if candidate_filter:
            pmra_e = row['pmra_error']
            pmdec_e = row['pmdec_error']
        else:
            pmra_e = 0
            pmdec_e = 0
        pmra_d = (pmra - cluster.pmra) / (2 * cluster.pmra_e + pmra_e)
        pmdec_d = (pmdec - cluster.pmdec) / (2 * cluster.pmdec_e + pmdec_e)
        pm_d = np.sqrt(pmra_d**2 + pmdec_d**2)
        return pm_d

    return pm_dist


def plx_distance(cluster, candidate_filter=False):
    cluster_plx_e = cluster.plx_bound

    def plx_dist(row):
        plx = row['parallax']

        if candidate_filter:
            plx_e = row['parallax_error']
        else:
            plx_e = 0
        plx_d = np.abs((plx - cluster.plx) / (cluster_plx_e / 3 + plx_e))
        return plx_d

    return plx_dist


def flux_error_to_mag_error(row, band, zero_point):
    mean_mag_error = -2.5 * row[f'phot_{band}_mean_flux_error'] / (row[f'phot_{band}_mean_flux'] * np.log(10))
    mean_mag_error = np.sqrt(mean_mag_error**2 + zero_point**2)
    return mean_mag_error


def phot_g_mean_mag_error():
    sigma_g_0 = 0.0027553202

    def phot_g_mean_mag(row):
        return flux_error_to_mag_error(row, 'g', sigma_g_0)
    return phot_g_mean_mag


def bp_rp_error():
    sigma_bp_0 = 0.0027901700
    sigma_rp_0 = 0.0037793818

    def bp_rp_err(row):
        bp_mean_mag_error = flux_error_to_mag_error(row, 'bp', sigma_bp_0)
        rp_mean_mag_error = flux_error_to_mag_error(row, 'rp', sigma_rp_0)
        return np.sqrt(bp_mean_mag_error**2 + rp_mean_mag_error**2)
    return bp_rp_err


def radius(cluster):
    def calculate_radius(row):
        x, y = projected_coordinates(row, cluster)
        return np.sqrt(x ** 2 + y ** 2)
    return calculate_radius


def pm_candidate_filter(cluster, candidate_filter_kwargs):
    pm_dist = pm_distance(cluster, candidate_filter=True)
    pm_d_threshold = candidate_filter_kwargs['pm_d_max']

    def candidate_filter(row):
        pm_d = pm_dist(row)

        candidate = False
        if pm_d < pm_d_threshold:
            candidate = True
        return candidate

    return candidate_filter


def plx_candidate_filter(cluster, candidates_filter_kwargs):
    plx_dist = plx_distance(cluster, candidate_filter=True)
    plx_d_threshold = candidates_filter_kwargs['plx_d_max']

    def candidate_filter(row):
        plx_d = plx_dist(row)

        candidate = False
        if plx_d < plx_d_threshold:
            candidate = True
        return candidate

    return candidate_filter


def isochrone_candidate_filter(isochrone, candidate_filter_kwargs):
    # isochrone_del = isochrone_delta(isochrone, candidate_filter_kwargs)

    def candidate_filter(row):
        # isochrone_d = isochrone_del(row)
        isochrone_d = np.array([row['bp_rp_d'], row['gmag_d']])

        candidate = False
        if norm(isochrone_d) < 1.0:
            candidate = True
        return candidate

    return candidate_filter


def make_candidate_filter(cluster, isochrone, candidate_filter_kwargs):
    filters = [plx_candidate_filter(cluster, candidate_filter_kwargs),
               pm_candidate_filter(cluster, candidate_filter_kwargs),
               isochrone_candidate_filter(isochrone, candidate_filter_kwargs)]

    def candidate_filter(row):
        candidate = True
        for filter_rule in filters:
            passed_filter = filter_rule(row)
            if not passed_filter:
                candidate = False
                break
        return candidate

    return candidate_filter


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
    for isochrone in isochrones:
        isochrone['Gmag'] += distance_modulus
        isochrone.rename(columns={'Gmag': 'phot_g_mean_mag'}, inplace=True)
        isochrone['G_BPmag'] += distance_modulus
        isochrone['G_RPmag'] += distance_modulus
        isochrone['bp_rp'] = isochrone['G_BPmag'] - isochrone['G_RPmag']
    return isochrones


def make_isochrone(isochrone_dir, age, dm):
    log_age_floor = int(age)
    isochrone_file = f'isochrones{log_age_floor}.dat'
    isochrone_path = os.path.join(isochrone_dir, isochrone_file)
    isochrones_df = pd.read_csv(isochrone_path, delim_whitespace=True, comment='#')

    ages = np.array(list(set(isochrones_df['logAge'].values)))
    closest_age = ages[np.argmin(np.abs(ages - age))]

    isochrone = isochrones_df[(isochrones_df['logAge'] == closest_age)].copy()

    isochrone = isochrone_correction([isochrone], dm)[0]
    isochrone = isochrone[isochrone['label'] <= 7]

    return isochrone


def extinction_correction(sources, cluster):
    extinction_g = danielski_g(cluster.a_v, sources['bp_rp'])
    extinction_bp = danielski_bp(cluster.a_v, sources['bp_rp'])
    extinction_rp = danielski_rp(cluster.a_v, sources['bp_rp'])

    sources['phot_g_mean_mag'] -= extinction_g
    sources['bp_rp'] -= extinction_bp - extinction_rp
    return sources


class Cluster:
    def __init__(self, parameters):
        self.name = parameters['Cluster'].values[0]
        self.ra = parameters['RA_ICRS'].values[0]
        self.dec = parameters['DE_ICRS'].values[0]
        self.r50 = parameters['r50'].values[0]
        self.pmra = parameters['pmRA*'].values[0]
        self.pmdec = parameters['pmDE'].values[0]
        self.plx = parameters['plx'].values[0]
        self.pmra_e = parameters['e_pmRA*'].values[0]
        self.pmdec_e = parameters['e_pmDE'].values[0]
        self.plx_e = parameters['e_plx'].values[0]
        self.age = parameters['AgeNN'].values[0]
        self.dm = parameters['DMNN'].values[0]
        self.a_v = parameters['AVNN'].values[0]
        self.dist = parameters['DistPc'].values[0]

        self.r_t = np.inf
        self.r_t_c = np.inf
        self.plx_bound = np.inf

    def add_tidal_radius(self, r_t):
        self.r_t = r_t

    def add_comparison_tidal_radius(self, r_t_c):
        self.r_t_c = r_t_c

    def add_plx_boundary(self, train_members, prob_threshold):
        members = train_members[prob_threshold <= train_members['PMemb']].copy()
        max_d = min(60, 5 * members.apply(radius(self), axis=1).std())
        max_plx_d = abs(self.plx - 1000 / (1000 / self.plx - max_d))
        std_of_plx_mean = self.plx_e / np.sqrt(len(train_members) - 1)
        plx_bound = max_plx_d + 3 * std_of_plx_mean
        print('Maximum radius', max_d, 'pc')
        print('Maximum plx range', max_plx_d, 'mas')
        print('Standard deviation of mean plx', std_of_plx_mean, 'mas')
        print('Cluster plx candidate range', plx_bound, 'mas')
        print(' ')
        self.plx_bound = plx_bound


def undo_split(df):
    ra = df['ra']
    if ra.max() - ra.min() > 180:
        ra = np.where(ra > 180, ra - 360, ra)
        df['ra'] = ra


def cone_set(cone_path, cluster):
    print('Loading cone data...', end=' ')
    t0 = time.time()

    cone = parse(cone_path)
    cone = cone.get_first_table().to_table(use_names_over_ids=True)
    cone = cone.to_pandas().dropna()

    extinction_correction(cone, cluster)
    undo_split(cone)

    cone['PMemb'] = 0

    print(f'Done in {time.time() - t0} sec')
    print('Cone sources', len(cone))
    print(' ')
    return cone


def member_set(members_path, cone):
    print(f'Loading members data...', end=' ')
    t0 = time.time()

    members_df = pd.read_csv(members_path)

    members_df = members_df[['Source', 'PMemb']].copy()
    members = pd.merge(members_df, cone, left_on='Source', right_on='source_id', how='inner', suffixes=('', '_y'),
                       copy=False)
    members = members.drop(['Source', 'PMemb_y'], axis=1)

    print(f'Done in {time.time() - t0} sec')
    print(f'Total members:', len(members))
    print(f'Missing members:', len(members_df) - len(members))
    print(' ')
    return members


def candidate_and_noise_set(cone, train_members, cluster, isochrone, candidate_filter_kwargs):
    print('Finding candidates...', end=' ')
    t0 = time.time()
    candidate_filter = make_candidate_filter(cluster, isochrone, candidate_filter_kwargs)
    candidate_indices = cone.apply(candidate_filter, axis=1)

    candidates = cone[candidate_indices].copy()
    noise = cone[~candidate_indices].copy()

    noise = noise[~noise['source_id'].isin(train_members['source_id'])].copy()

    # if len(all_noise) > max_noise:
    #     noise = all_noise.sample(max_noise)
    # else:
    #     noise = all_noise
    print(f'Done in {time.time() - t0} sec')
    print('Candidates:', len(candidates))
    print('Noise:', len(noise))
    print(' ')
    return candidates, noise


# def add_fields(sources, cluster, isochrone, candidate_filter_kwargs):
#     field_fs = [pm_distance(cluster), plx_distance(cluster), radius(cluster), bp_rp_error(),
#                 phot_g_mean_mag_error(), isochrone_delta(isochrone, candidate_filter_kwargs)]
#     field_labels = ['pm_d', 'plx_d', 'r', 'bp_rp_error', 'phot_g_mean_mag_error', ['bp_rp_d', 'gmag_d']]
#     for f, label in zip(field_fs, field_labels):
#         sources.add_field(f, label)


def add_fields(cone, train_members, comparison_members, cluster, isochrone, candidate_filter_kwargs):
    print('Loading cone data...', end=' ')
    t0 = time.time()
    field_fs = [pm_distance(cluster), plx_distance(cluster), radius(cluster), bp_rp_error(),
                phot_g_mean_mag_error(), isochrone_delta(isochrone, candidate_filter_kwargs)]
    field_labels = ['pm_d', 'plx_d', 'r', 'bp_rp_error', 'phot_g_mean_mag_error', ['bp_rp_d', 'gmag_d']]
    for f, label in zip(field_fs, field_labels):
        for subset in [cone, train_members, comparison_members]:
            result_type = None
            if type(label) == list:
                result_type = 'expand'
            subset[label] = subset.apply(f, axis=1, result_type=result_type)
    print(f'Done in {time.time() - t0} sec')
    print(' ')


def parse_sources(cone_path, train_members_path, comparison_members_path, cluster, isochrone, probability_threshold,
                  candidate_filter_kwargs):
    train_ref = members_path_to_ref(train_members_path)
    if not os.path.exists(train_members_path):
        raise ValueError(f'No members available for cluster {cluster.name} in the {train_ref} members!')

    comparison_ref = members_path_to_ref(comparison_members_path)
    if not os.path.exists(comparison_members_path):
        comparison_members_path = train_members_path
        comparison_ref = train_ref
        print(f'No members available for cluster {cluster.name} in the {comparison_ref} members. '
              f'Using the {train_ref} members as comparison instead')

    cone = cone_set(cone_path, cluster)
    train_members = member_set(train_members_path, cone)
    comparison_members = member_set(comparison_members_path, cone)

    cluster.add_plx_boundary(train_members, probability_threshold)
    add_fields(cone, train_members, comparison_members, cluster, isochrone, candidate_filter_kwargs)

    candidates, noise = candidate_and_noise_set(cone, train_members, cluster, isochrone, candidate_filter_kwargs)

    # sample_noise = noise.sample(max_noise)
    # plot_noise = noise.copy()

    sources = Sources(train_members,
                      comparison_members,
                      candidates,
                      noise,
                      probability_threshold,
                      train_ref,
                      comparison_ref)

    # add_fields(sources, cluster, isochrone, candidate_filter_kwargs)

    return sources
