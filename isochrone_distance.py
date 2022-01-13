from filter_cone_data import point_line_section_delta, isochrone_distances, isochrone_distance_filter, load_cone_data, \
    load_cluster_data, make_isochrone, make_hp_member_df, make_field_df
import matplotlib.pyplot as plt
import numpy as np
import time


def norm(vec):
    return np.sqrt(np.sum(np.square(vec)))


x_ = np.random.uniform(0, 1, 10000)
y_ = np.random.uniform(0, 1, 10000)

points = np.stack((x_, y_)).T
line = np.array([[0.25, 0.25], [0.75, 0.75]])
subset = np.array([norm(point_line_section_delta(point, line[0], line[1])) > 0.1 for point in points])

plt.scatter(x_[np.where(subset)[0]], y_[np.where(subset)[0]])
plt.plot(line[:, 0], line[:, 1])
plt.show()

data_dir = 'practice_data'
isochrone_file = 'isochrone.dat'
cone_file = 'NGC_2509_cone.csv'
cluster_file = 'NGC_2509.tsv'
probability_threshold = 0.7
candidate_filter_kwargs = {'plx_sigma': 3.0,
                           'gmag_max_d': 0.6,
                           'pm_max_d': 1.0,
                           'bp_rp_max_d': 0.2}

cone_df = load_cone_data(data_dir, cone_file)
cluster_df = load_cluster_data(data_dir, cluster_file)

hp_members = make_hp_member_df(cone_df, cluster_df, probability_threshold)
field = make_field_df(cone_df, hp_members)

mean_plx = np.mean(hp_members['parallax'].values)
isochrone = make_isochrone(data_dir, isochrone_file, mean_plx, extinction_v=0.13, cutoff=135)

t0 = time.time()

fast = True
if fast:
    distances = isochrone_distances(field,
                                    isochrone,
                                    candidate_filter_kwargs['bp_rp_max_d'],
                                    candidate_filter_kwargs['gmag_max_d'])
    candidates = field[distances < 1]
else:
    rule = isochrone_distance_filter(isochrone,
                                     candidate_filter_kwargs['bp_rp_max_d'],
                                     candidate_filter_kwargs['gmag_max_d'])
    candidates = field[field.apply(rule, axis=1)]

print('Time required for finding candidates:', time.time() - t0)

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(isochrone['bp_rp'], isochrone['phot_g_mean_mag'], rasterized=True)
ax.scatter(candidates['bp_rp'], candidates['phot_g_mean_mag'], c='grey', s=0.1, rasterized=True)
ax.scatter(hp_members['bp_rp'], hp_members['phot_g_mean_mag'], c='blue', s=10.0, rasterized=True)

ax.set_ylim(6, 22)
ax.set_xlim(-0.5, 2.5)
ax.invert_yaxis()
plt.show()
