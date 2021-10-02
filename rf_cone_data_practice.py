import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def rf_practice_with_cone(data_dir,
                          cone_file,
                          cluster_file,
                          probability_threshold,
                          train_columns,
                          test_columns,
                          cluster_name='cluster',
                          explore_max_depth_space=True,
                          show_prediction=False):
    # load practice data
    practice_data_path = data_dir
    practice_data_cone_file = os.path.join(practice_data_path, cone_file)
    practice_data_cluster_file = os.path.join(practice_data_path, cluster_file)

    # create cone dataframe
    cone_df = pd.read_csv(practice_data_cone_file)

    # create cluster dataframe
    cluster_df = pd.read_csv(practice_data_cluster_file, sep='\t', header=61)
    cluster_df = cluster_df.iloc[2:]
    cluster_df = cluster_df.reset_index(drop=True)

    print(f'Number of sources in cone: {len(cone_df)}')
    print(f'Number of sources in cluster: {len(cluster_df)}\n')

    # drop the cluster sources below the threshold from the cone
    test_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) <
                                         probability_threshold)]['Source'].astype(np.int64).values
    test_member_indices = cone_df[(cone_df.source_id.isin(test_member_source_ids))].index
    cone_df.drop(test_member_indices, inplace=True)

    # make a new column in the cone dataframe that indicates whether a source belongs to the cluster
    train_member_source_ids = cluster_df[(cluster_df['PMemb'].astype(np.float32) >=
                                          probability_threshold)]['Source'].astype(np.int64).values
    all_source_ids = cone_df['source_id'].values
    cone_df['cluster'] = np.array([int(source_id in train_member_source_ids) for source_id in all_source_ids])

    print(f'{len(train_member_source_ids)} '
          f'cluster sources were selected with a membership probability of >= {probability_threshold}')
    print(f'The {len(test_member_source_ids)} remaining cluster sources were removed from the cone, '
          f'which now contains {len(all_source_ids)} sources')

    # new dataframes with only members and non members
    members = cone_df[(cone_df['cluster'] == 1)]
    non_members = cone_df[(cone_df['cluster'] == 0)]

    alpha_non_members = 0.5

    # plot members and non members
    if show_prediction or explore_max_depth_space:
        horz_plots = 3
        fig_width = 18
    else:
        horz_plots = 2
        fig_width = 12
    fig, ax = plt.subplots(2, horz_plots, figsize=(fig_width, 12))
    ax[0, 0].scatter(non_members['ra'], non_members['DEC'],
                     label='non members', s=0.1, c='gray', alpha=alpha_non_members)
    ax[0, 0].scatter(members['ra'], members['DEC'],
                     label=f'members (>= {probability_threshold})', s=1.0, c='red')
    ax[0, 0].set_title(cluster_name)
    ax[0, 0].set_xlabel('RA')
    ax[0, 0].set_ylabel('DEC')
    ax[0, 0].legend()
    ax[1, 0].scatter(non_members['bp_rp'], non_members['phot_g_mean_mag'],
                     label='non members', s=0.1, c='gray', alpha=alpha_non_members)
    ax[1, 0].scatter(members['bp_rp'], members['phot_g_mean_mag'],
                     label=f'members (>= {probability_threshold})', s=1.0, c='red')
    ax[1, 0].set_title(cluster_name)
    ax[1, 0].set_xlabel('bp_rp')
    ax[1, 0].set_ylabel('phot_g_mean_mag')
    ax[1, 0].legend()
    ax[0, 1].scatter(non_members['pmra'], non_members['pmdec'],
                     label='non members', s=0.1, c='gray', alpha=alpha_non_members)
    ax[0, 1].scatter(members['pmra'], members['pmdec'],
                     label=f'members (>= {probability_threshold})', s=1.0, c='red')
    ax[0, 1].set_title(cluster_name)
    ax[0, 1].set_xlabel('pmra')
    ax[0, 1].set_ylabel('pmdec')
    ax[0, 1].set_xlim(-10, 10)
    ax[0, 1].set_ylim(-10, 10)
    ax[0, 1].legend()
    ax[1, 1].scatter(non_members['phot_g_mean_mag'], non_members['parallax'],
                     label='non members', s=0.1, c='gray', alpha=alpha_non_members)
    ax[1, 1].scatter(members['phot_g_mean_mag'], members['parallax'],
                     label=f'members (>= {probability_threshold})', s=1.0, c='red')
    ax[1, 1].set_title(cluster_name)
    ax[1, 1].set_xlabel('phot_g_mean_mag')
    ax[1, 1].set_ylabel('parallax')
    ax[1, 1].set_ylim(-10, 10)
    ax[1, 1].legend()
    if show_prediction or explore_max_depth_space:
        # create training data for random forest
        train_data = cone_df[train_columns + ['cluster']].dropna().astype(np.float32)
        print(f'{len(cone_df) - len(train_data)} cone sources were removed, because their data were incomplete\n')
        x_train = np.array(train_data.drop('cluster', axis=1))
        y_train = np.array(train_data['cluster'], dtype=np.int32)
        n_train = x_train.shape[0]
        print(f'Training set size: {n_train}')

        # create test data for random forest
        test_cluster_df = cluster_df[(cluster_df['Source'].astype(np.int64).isin(test_member_source_ids))]
        test_data = test_cluster_df[test_columns].astype(np.float32)
        x_test = np.array(test_data)
        y_test = np.ones(len(x_test), dtype=np.int32)
        n_test = x_test.shape[0]
        print(f'Test set size: {n_test}\n')

        best_max_depth = 10

        if show_prediction:
            # create and train random forest with best max depth
            best_clf = RandomForestClassifier(max_depth=best_max_depth, random_state=0)
            best_clf.fit(x_train, y_train)

            # list the probabilities of the sources
            # which are selected by the random forest as either member or non member
            rf_member_probs = []
            rf_non_member_probs = []

            for prob, rf_member in zip(test_cluster_df['PMemb'].astype(np.float32).values, best_clf.predict(x_test)):
                if rf_member == 1:
                    rf_member_probs.append(prob)
                else:
                    rf_non_member_probs.append(prob)

            # make a histogram of the rf selected members/non members
            # with bins corresponding to their database probability
            ax[0, 2].hist([rf_non_member_probs, rf_member_probs], int(probability_threshold * 10) - 1,
                          label=['rf predicted non members', 'rf predicted members'], stacked=True)
            ax[0, 2].set_title('Database membership probability of rf (non) members')
            ax[0, 2].set_xlabel('database membership probability')
            ax[0, 2].set_ylabel('counts')
            ax[0, 2].legend()
            if not explore_max_depth_space:
                fig.delaxes(ax[1, 2])
        if explore_max_depth_space:
            # explore for varying max depths how well a random forest performs on the test dataset
            train_scores = []
            test_scores = []

            max_depths = np.arange(1, 20)
            best_test_score = 0

            for max_depth in max_depths:
                # create random forest
                clf = RandomForestClassifier(max_depth=max_depth, random_state=0)

                # fit random forest to training data
                clf.fit(x_train, y_train)

                # calculate and print scores
                train_score = clf.score(x_train, y_train)
                test_score = clf.score(x_test, y_test)
                train_scores.append(train_score)
                test_scores.append(test_score)

                if test_score > best_test_score:
                    best_test_score = test_score
                    best_max_depth = max_depth

            print(f'Random forest best score on training data: {max(train_scores)}')
            print(f'Random forest best score on test data: {best_test_score}')
            print(f'Random forest best max depth: {best_max_depth}\n')

            ax[1, 2].plot(max_depths, test_scores, label='test')
            ax[1, 2].plot(max_depths, train_scores, label='train')
            ax[1, 2].set_title('Random forest max depth optimization')
            ax[1, 2].set_xlabel('max depth')
            ax[1, 2].set_ylabel('score')
            ax[1, 2].set_ylim(0, 1.2)
            ax[1, 2].legend()
            if not explore_max_depth_space:
                fig.delaxes(ax[0, 2])
    plt.show()


if __name__ == "__main__":
    # lists of columns to use for training the random forest
    # the cone and cluster data have slightly different names for their columns
    tgas_cone_columns = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec']
    tgas_and_photometric_cone_columns = ['ra', 'DEC', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']

    train_col = tgas_and_photometric_cone_columns

    tgas_cluster_columns = ['RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE']
    tgas_and_photometric_cluster_columns = ['RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE', 'Gmag', 'BP-RP']

    test_col = tgas_and_photometric_cluster_columns

    rf_practice_with_cone(data_dir='practice_data',
                          cone_file='NGC_2509_cone.csv',
                          cluster_file='NGC_2509.tsv',
                          probability_threshold=0.9,
                          train_columns=train_col,
                          test_columns=test_col,
                          cluster_name='NCG_2509',
                          explore_max_depth_space=False,
                          show_prediction=True)
