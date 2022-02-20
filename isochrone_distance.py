from filter_cone_data import isochrone_distance_filter, load_cone_data, \
    load_cluster_data, make_isochrone, make_member_df, make_field_df, most_likely_value, \
    source_isochrone_section_delta, get_cluster_parameters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os


def norm(vec):
    return np.sqrt(np.sum(np.square(vec)))


cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')

cluster_name = 'NGC_2509'
# cluster_name = 'Ruprecht_33'
# cluster_name = 'IC_2714'
# cluster_name = 'Ruprecht_135'

cone_path = os.path.join(data_dir, 'cones', cluster_name + '.csv')
cluster_path = os.path.join(data_dir, 'members', cluster_name + '.csv')
isochrone_path = os.path.join(data_dir, 'isochrones', 'isochrones.dat')
cluster_params_path = os.path.join(data_dir, 'cluster_parameters.tsv')

cluster_kwargs = get_cluster_parameters(cluster_params_path, cluster_name)

probability_threshold = 0.7
candidate_filter_kwargs = {'plx_sigma': 3.0,
                           'gmag_max_d': 0.5,
                           'pm_max_d': 1.0,
                           'bp_rp_max_d': 0.2}

cone_df = load_cone_data(cone_path, chunksize=None)
cluster_df = load_cluster_data(cluster_path)

# hp_members = make_member_df(cone_df, cluster_df, probability_threshold)
# field = make_field_df(cone_df, hp_members)
#
# ages = np.arange(8, 10, 0.10)
#
# isochrone = make_isochrone(isochrone_path, age=cluster_kwargs['age'], z=0.015, dm=cluster_kwargs['dm'],
#                            extinction_v=cluster_kwargs['a_v'])
#
# isochrone_pairs = np.stack((np.stack((isochrone['bp_rp'].values[:-1],
#                                       isochrone['bp_rp'].values[1:]), axis=-1),
#                             np.stack((isochrone['phot_g_mean_mag'].values[:-1],
#                                       isochrone['phot_g_mean_mag'].values[1:]), axis=-1)), axis=-1)

# t0 = time.time()
#
# rule = isochrone_distance_filter(isochrone_pairs,
#                                  candidate_filter_kwargs['bp_rp_max_d'],
#                                  candidate_filter_kwargs['gmag_max_d'])
# candidates = field[field.apply(rule, axis=1)].copy()
#
# print('Time required for finding candidates:', time.time() - t0)
#
# def isochrone_delta(row):
#     bp_rp, gmag = row['bp_rp'], row['phot_g_mean_mag']
#     point = np.array([bp_rp, gmag])
#
#     delta = source_isochrone_section_delta(point, isochrone_pairs,
#                                            candidate_filter_kwargs['bp_rp_max_d'],
#                                            candidate_filter_kwargs['gmag_max_d'])
#     return delta
#
#
# candidates[['isochrone_bp_rp_d', 'isochrone_gmag_d']] = candidates.apply(isochrone_delta, axis=1, result_type='expand')
#
# print('Time required for calculating isochrone distances:', time.time() - t0)
#
# non_members = pd.concat((field, candidates)).drop_duplicates(subset='source_id', keep=False)

gmag_max = 22.0
x = np.linspace(6, gmag_max - 0.7, 100)


def f(x):
    y = 2.0 / (-(x - gmag_max))**2.3 + 0.2
    # y = 2 * np.exp(0.5 * (x - 1.1 * gmag_max))
    return y


# def h(x):
#     y = 2.0 / (-(x - 21.0))**2.3
#     # y = 2 * np.exp(0.5 * (x - 1.1 * gmag_max))
#     return y

fig, ax = plt.subplots(figsize=(8, 8))

# for age in ages:
#     isochrone = make_isochrone(isochrone_path, age=age, z=0.015, dm=cluster_kwargs['dm'],
#                                extinction_v=cluster_kwargs['a_v'])
#     ax.plot(isochrone['bp_rp'], isochrone['phot_g_mean_mag'], rasterized=True)

# ax.plot(isochrone['bp_rp'], isochrone['phot_g_mean_mag'], rasterized=True)
ax.scatter(cone_df['parallax'], np.abs(cone_df['parallax_error']), s=0.1, rasterized=True)
ax.plot(x, f(x), c='red')
# ax.scatter(candidates['bp_rp'], candidates['phot_g_mean_mag'], c='red', s=0.5, rasterized=True)
# ax.scatter(non_members['bp_rp'], non_members['phot_g_mean_mag'], c='grey', s=0.1, rasterized=True)
# ax.scatter(hp_members['bp_rp'], hp_members['phot_g_mean_mag'], c='blue', s=10.0, rasterized=True)

# ax.set_ylim(5, 35)
# ax.set_xlim(-1, 8)

# ax.set_ylim(16, 21)
# ax.set_xlim(0.75, 2.5)

# ax.set_ylim(11, 21)
# ax.set_xlim(-1, 3.0)
# ax.invert_yaxis()
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlabel('phot_g_mean_mag')
ax.set_ylabel('parallax error')
plt.show()
