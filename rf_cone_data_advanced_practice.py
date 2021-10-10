import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from filter_cone_data import divide_cone


def extinction_correction(isochrones, member_df, extinction_v):
    extinction_g = 0.83627 * extinction_v
    mean_plx = np.mean(member_df['parallax'].values)  # in mas
    mean_d = 1000 / mean_plx
    gmag_correction = 5 * np.log10(mean_d) - 5 - extinction_g
    bp_rp_correction = (1.08337 - 0.6343) * extinction_v

    for isochrone in isochrones:
        isochrone['bp_rp'] = isochrone['G_BPmag'] - isochrone['G_RPmag'] + bp_rp_correction
        isochrone['Gmag'] += gmag_correction

    return isochrones


def rf_advanced_practice_with_cone(data_dir,
                                   cone_file,
                                   cluster_file,
                                   isochrone_file,
                                   probability_threshold,
                                   train_columns,
                                   plx_sigma_factor=1.0,
                                   pm_sigma_factor=1.0,
                                   isochrone_sigma_factor=0.3,
                                   cluster_name='cluster',
                                   drop_database_candidates=False,
                                   plot_predicted_members=True,
                                   plot_predicted_non_members=True):
    # load practice data
    practice_data_path = data_dir
    practice_data_cone_file = os.path.join(practice_data_path, cone_file)
    practice_data_cluster_file = os.path.join(practice_data_path, cluster_file)
    practice_data_isochrone_file = os.path.join(practice_data_path, isochrone_file)

    # create cone dataframe
    cone_df = pd.read_csv(practice_data_cone_file)

    # create cluster dataframe
    cluster_df = pd.read_csv(practice_data_cluster_file, sep='\t', header=61)
    cluster_df = cluster_df.iloc[2:]
    cluster_df = cluster_df.reset_index(drop=True)

    # create isochrone dataframe
    isochrone_df = pd.read_csv(practice_data_isochrone_file, sep='\s+', comment='#')

    isochrones = []
    for age in list(set(isochrone_df['logAge'].values)):
        isochrones.append(isochrone_df[(isochrone_df['logAge'] == age)].iloc[:135])

    members, database_candidates, candidates, non_members = divide_cone(cone_df,
                                                                        cluster_df,
                                                                        train_columns,
                                                                        probability_threshold,
                                                                        plx_sigma_factor,
                                                                        pm_sigma_factor,
                                                                        isochrone_sigma_factor,
                                                                        drop_database_candidates=drop_database_candidates)

    if len(candidates) > 0:
        alpha_non_members = 0.8
        dot_size = 3.0
        isochrones = extinction_correction(isochrones, members, extinction_v=0.15)

        # create training data for random forest
        train_data = pd.concat([members, non_members], ignore_index=True)
        x_train = train_data.values
        y_train = np.concatenate((np.ones(len(members)), np.zeros(len(non_members))))
        n_train = x_train.shape[0]
        print(f'\nTraining set size: {n_train}')

        # create test data for random forest
        test_data = candidates
        x_test = test_data.values
        n_test = x_test.shape[0]
        print(f'Test set size: {n_test}')

        max_depth = 10

        clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        candidates['pred'] = predictions

        predicted_members = candidates[(candidates['pred'] == 1)]
        predicted_non_members = candidates[(candidates['pred'] == 0)]

        n_duplicates = sum(database_candidates.append(predicted_members.drop('pred', axis=1)).duplicated().values.astype(np.int32))

        print(f'\n{len(predicted_members)}/{len(candidates)} of the candidates were predicted to be members')
        print(f'{n_duplicates}/{len(database_candidates)} predicted members are also present in the database candidates')
        print(f'{len(predicted_non_members)}/{len(candidates)} of the candidates were predicted to be non members')

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax[0, 0].scatter(non_members['ra'], non_members['DEC'],
                         label='non members', s=0.1, c='gray', alpha=alpha_non_members)
        ax[0, 0].scatter(members['ra'], members['DEC'],
                         label=f'members (>= {probability_threshold})', s=dot_size, c='red')
        if plot_predicted_non_members:
            ax[0, 0].scatter(predicted_non_members['ra'], predicted_non_members['DEC'],
                             label=f'predicted_non_members', s=dot_size, c='purple')
        if plot_predicted_members:
            ax[0, 0].scatter(predicted_members['ra'], predicted_members['DEC'],
                             label=f'predicted_members', s=dot_size, c='blue')
        ax[0, 0].set_title(cluster_name)
        ax[0, 0].set_xlabel('RA')
        ax[0, 0].set_ylabel('DEC')
        ax[0, 0].legend()

        ax[1, 0].scatter(non_members['bp_rp'], non_members['phot_g_mean_mag'],
                         label='non members', s=0.1, c='gray', alpha=alpha_non_members)
        ax[1, 0].scatter(members['bp_rp'], members['phot_g_mean_mag'],
                         label=f'members (>= {probability_threshold})', s=dot_size, c='red')
        if plot_predicted_non_members:
            ax[1, 0].scatter(predicted_non_members['bp_rp'], predicted_non_members['phot_g_mean_mag'],
                             label=f'predicted_non_members', s=dot_size, c='purple')
        if plot_predicted_members:
            ax[1, 0].scatter(predicted_members['bp_rp'], predicted_members['phot_g_mean_mag'],
                             label=f'predicted_members', s=dot_size, c='blue')
        for isochrone in [isochrones[4]]:
            ax[1, 0].plot(isochrone['bp_rp'], isochrone['Gmag'])  # , label=f'{isochrone["logAge"].values[0]}')
        ax[1, 0].set_title(cluster_name)
        ax[1, 0].set_xlim(-0.5, 2.5)
        ax[1, 0].set_ylim(10, 20)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlabel('bp_rp')
        ax[1, 0].set_ylabel('phot_g_mean_mag')
        ax[1, 0].legend()

        ax[0, 1].scatter(non_members['pmra'], non_members['pmdec'],
                         label='non members', s=0.1, c='gray', alpha=alpha_non_members)
        ax[0, 1].scatter(members['pmra'], members['pmdec'],
                         label=f'members (>= {probability_threshold})', s=dot_size, c='red')
        if plot_predicted_non_members:
            ax[0, 1].scatter(predicted_non_members['pmra'], predicted_non_members['pmdec'],
                             label=f'predicted_non_members', s=dot_size, c='purple')
        if plot_predicted_members:
            ax[0, 1].scatter(predicted_members['pmra'], predicted_members['pmdec'],
                             label=f'predicted_members', s=dot_size, c='blue')
        ax[0, 1].set_title(cluster_name)
        ax[0, 1].set_xlabel('pmra')
        ax[0, 1].set_ylabel('pmdec')
        ax[0, 1].set_xlim(-4.6, -0.6)
        ax[0, 1].set_ylim(-1.2, 2.8)
        ax[0, 1].legend()

        ax[1, 1].scatter(non_members['phot_g_mean_mag'], non_members['parallax'],
                         label='non members', s=0.1, c='gray', alpha=alpha_non_members)
        ax[1, 1].scatter(members['phot_g_mean_mag'], members['parallax'],
                         label=f'members (>= {probability_threshold})', s=dot_size, c='red')
        if plot_predicted_non_members:
            ax[1, 1].scatter(predicted_non_members['phot_g_mean_mag'], predicted_non_members['parallax'],
                             label=f'predicted_non_members', s=dot_size, c='purple')
        if plot_predicted_members:
            ax[1, 1].scatter(predicted_members['phot_g_mean_mag'], predicted_members['parallax'],
                             label=f'predicted_members', s=dot_size, c='blue')
        ax[1, 1].set_title(cluster_name)
        ax[1, 1].set_xlabel('phot_g_mean_mag')
        ax[1, 1].set_ylabel('parallax')
        ax[1, 1].set_ylim(-1.0, 1.0)
        ax[1, 1].legend()
        plt.show()
    else:
        print('No candidates were found!')


if __name__ == "__main__":
    # lists of columns to use for training the random forest
    tgas_cone_columns = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec']
    tgas_and_photometric_cone_columns = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp', 'ruwe']

    train_col = tgas_and_photometric_cone_columns

    rf_advanced_practice_with_cone(data_dir='practice_data',
                                   cone_file='NGC_2509_cone.csv',
                                   cluster_file='NGC_2509.tsv',
                                   isochrone_file='isochrone.dat',
                                   probability_threshold=0.9,
                                   train_columns=train_col,
                                   plx_sigma_factor=0.3,
                                   pm_sigma_factor=1.0,
                                   isochrone_sigma_factor=0.4,
                                   cluster_name='NCG_2509',
                                   drop_database_candidates=False,
                                   plot_predicted_members=True,
                                   plot_predicted_non_members=False)
