import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from filter_cone_data import divide_cone, load_cone_data, load_cluster_data


def isochrone_correction(isochrones, distance, extinction_v):
    extinction_g = 0.83627 * extinction_v
    extinction_bp = 1.08337 * extinction_v
    extinction_rp = 0.6343 * extinction_v

    distance_modulus = 5 * np.log10(distance) - 5

    gmag_correction = distance_modulus + extinction_g
    bpmag_correction = distance_modulus + extinction_bp
    rpmag_correction = distance_modulus + extinction_rp

    for isochrone in isochrones:
        isochrone['Gmag'] += gmag_correction
        isochrone['G_BPmag'] += bpmag_correction
        isochrone['G_RPmag'] += rpmag_correction
        isochrone['bp_rp'] = isochrone['G_BPmag'] - isochrone['G_RPmag']

    return isochrones


def duplicates(df1, df2):
    combined = df1.append(df2)
    return combined[combined.duplicated(keep='last')]


def load_isochrone_data(data_dir, isochrone_file):
    practice_data_isochrone_file = os.path.join(data_dir, isochrone_file)

    # create isochrones dataframe
    isochrones_df = pd.read_csv(practice_data_isochrone_file, delim_whitespace=True, comment='#')
    return isochrones_df


def rf_advanced_practice_with_cone(data_dir,
                                   cone_file,
                                   cluster_file,
                                   isochrone_file,
                                   probability_threshold,
                                   candidate_selection_columns,
                                   train_columns,
                                   plx_sigma=3.0,
                                   gmag_max_d=1.0,
                                   pm_max_d=1.0,
                                   bp_rp_max_d=1.0,
                                   extinction_v=0.15,
                                   cluster_name='cluster',
                                   drop_db_candidates=False,
                                   plot_members=True,
                                   plot_non_members=True,
                                   plot_predicted_members=True,
                                   plot_predicted_non_members=True,
                                   plot_plx_error_bars=False):
    cone_df = load_cone_data(data_dir, cone_file)
    cluster_df = load_cluster_data(data_dir, cluster_file)
    isochrones_df = load_isochrone_data(data_dir, isochrone_file)

    # divide isochrones dataframe in a separate dataframe for each isochrone
    isochrones = []
    for age in list(set(isochrones_df['logAge'].values)):
        isochrones.append(isochrones_df[(isochrones_df['logAge'] == age)].iloc[:135])

    # isochrones = [isochrones[5]]

    # divide the cone in high probability members, low probability members, candidates and non members
    high_prob_members, low_prob_members, candidates, non_members = divide_cone(cone_df,
                                                                               cluster_df,
                                                                               candidate_selection_columns,
                                                                               probability_threshold,
                                                                               plx_sigma,
                                                                               gmag_max_d,
                                                                               pm_max_d,
                                                                               bp_rp_max_d,
                                                                               drop_db_candidates=drop_db_candidates)

    high_prob_members_plx_errors = high_prob_members['parallax_error'].values

    high_prob_members, low_prob_members, candidates, non_members = [df[train_columns] for df in [high_prob_members,
                                                                                                 low_prob_members,
                                                                                                 candidates,
                                                                                                 non_members]]

    if len(candidates) > 0:
        # correct the isochrones for distance and extinction
        mean_plx = np.mean(high_prob_members['parallax'].values)  # in mas
        mean_distance = 1000 / mean_plx
        isochrone_distance = 0.80 * mean_distance

        # cheating with distance and extinction to make isochrone fit
        isochrones = isochrone_correction(isochrones, isochrone_distance, extinction_v=extinction_v)

        print('\nMean distance of isochrone: ', isochrone_distance, ' pc')
        print('A_V of the isochrone: ', extinction_v)
        print('Log age of the isochrone: ', isochrones[0]['logAge'].values[0])

        # create training data for random forest
        train_data = pd.concat([high_prob_members, non_members], ignore_index=True)
        x_train = train_data.values
        y_train = np.concatenate((np.ones(len(high_prob_members)), np.zeros(len(non_members))))
        n_train = x_train.shape[0]
        print(f'\nTraining set size: {n_train}')

        # create test data for random forest
        x_test = candidates.values
        n_test = x_test.shape[0]
        print(f'Test set size: {n_test}')

        # train random forest with high probability members and non members
        max_depth = 10
        clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
        clf.fit(x_train, y_train)

        # predict membership of candidates
        predictions = clf.predict(x_test).astype(bool)

        # make subsets from candidates based on membership prediction
        predicted_members = candidates[predictions]
        predicted_non_members = candidates[np.invert(predictions)]

        print(f'\n{len(predicted_members)}/{len(candidates)} of the candidates were predicted to be members')
        print(f'{len(predicted_non_members)}/{len(candidates)} of the candidates were predicted to be non members')

        # see how the low probability members were treated by the random forest
        if not drop_db_candidates:
            low_prob_members_in_predicted_members = duplicates(low_prob_members, predicted_members)
            low_prob_members_in_predicted_non_members = duplicates(low_prob_members, predicted_non_members)
            low_prob_members_in_field = duplicates(low_prob_members, non_members)

            n_low_prob_members_in_predicted_members = len(low_prob_members_in_predicted_members)
            n_low_prob_members_in_predicted_non_members = len(low_prob_members_in_predicted_non_members)
            n_low_prob_members_in_field = len(low_prob_members_in_field)
            n_low_prob_members_candidates = len(low_prob_members)

            print(f'\n{n_low_prob_members_in_predicted_members}/{n_low_prob_members_candidates} '
                  f'of low probability members were predicted to be members')
            print(f'{n_low_prob_members_in_predicted_non_members}/{n_low_prob_members_candidates} '
                  f'of low probability members were predicted to be non members')
            print(f'{n_low_prob_members_in_field}/{n_low_prob_members_candidates} '
                  f'of low probability members were not selected as candidates')

        # plot results
        fig, ax = plt.subplots(2, 2, figsize=(16, 16))
        alpha_non_members = 0.8
        dot_size = 3.0

        def plot_func(axis, x, y):
            if plot_non_members:
                axis.scatter(non_members[x], non_members[y], label='non members', s=0.1 * dot_size, c='gray',
                             alpha=alpha_non_members, rasterized=True)
                axis.scatter(low_prob_members_in_field[x], low_prob_members_in_field[y],
                             marker='+', label=f'non_members_in_db', s=10 * dot_size, c='gray',
                             rasterized=True)
            if plot_predicted_non_members:
                axis.scatter(predicted_non_members[x], predicted_non_members[y], label=f'predicted_non_members',
                             s=dot_size, c='red', rasterized=True)
                axis.scatter(low_prob_members_in_predicted_non_members[x], low_prob_members_in_predicted_non_members[y],
                             marker='+', label=f'predicted_non_members_in_db', s=10 * dot_size, c='red',
                             rasterized=True)
            if plot_members:
                axis.scatter(high_prob_members[x], high_prob_members[y],
                             label=f'members (>= {probability_threshold})', s=dot_size, c='blue', rasterized=True)
            if plot_predicted_members:
                axis.scatter(predicted_members[x], predicted_members[y], label=f'predicted_members', s=dot_size,
                             c='green', rasterized=True)
                axis.scatter(low_prob_members_in_predicted_members[x], low_prob_members_in_predicted_members[y],
                             marker='+', label=f'predicted_members_in_db', s=10 * dot_size, c='green', rasterized=True)
            axis.set_title(cluster_name)
            axis.set_xlabel(x)
            axis.set_ylabel(y)
            axis.legend(fontsize='small')

        plot_func(ax[0, 0], 'ra', 'DEC')

        plot_func(ax[1, 0], 'bp_rp', 'phot_g_mean_mag')
        for isochrone in isochrones:
            ax[1, 0].plot(isochrone['bp_rp'], isochrone['Gmag'])  # , label=f'{isochrone["logAge"].values[0]}')
        ax[1, 0].set_xlim(0.25, 2.0)
        ax[1, 0].set_ylim(10.5, 18.5)
        ax[1, 0].invert_yaxis()

        plot_func(ax[0, 1], 'pmra', 'pmdec')
        ax[0, 1].set_xlim(-3.6, -1.6)
        ax[0, 1].set_ylim(-0.2, 1.8)

        plot_func(ax[1, 1], 'phot_g_mean_mag', 'parallax')
        if plot_plx_error_bars:
            ax[1, 1].errorbar(high_prob_members['phot_g_mean_mag'], high_prob_members['parallax'],
                              yerr=high_prob_members_plx_errors, fmt='.', c='blue', rasterized=True)
        ax[1, 1].set_ylim(-0.5, 1.25)
        ax[1, 1].set_xlim(10.5, 18.5)
        plt.savefig('results/NGC_2509_plots.pdf')
        plt.show()
    else:
        print('No candidates were found!')


if __name__ == "__main__":
    # lists of columns to use for training the random forest
    train_columns_small = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec']
    train_columns_big = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp', 'ruwe']
    candidate_select_columns = ['ra', 'DEC', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp',
                                'ruwe']

    rf_advanced_practice_with_cone(data_dir='practice_data',
                                   cone_file='NGC_2509_cone.csv',
                                   cluster_file='NGC_2509.tsv',
                                   isochrone_file='isochrone.dat',
                                   probability_threshold=0.9,
                                   candidate_selection_columns=candidate_select_columns,
                                   train_columns=train_columns_big,
                                   plx_sigma=3.0,
                                   gmag_max_d=0.2,
                                   pm_max_d=0.5,
                                   bp_rp_max_d=0.05,
                                   extinction_v=0.15,
                                   cluster_name='NCG_2509',
                                   plot_members=True,
                                   plot_non_members=True,
                                   plot_predicted_members=True,
                                   plot_predicted_non_members=True)