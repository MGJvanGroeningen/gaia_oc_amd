import pandas as pd
from filter_cone_data import parse_data
from data_generation import generate_rf_input
from visualization import make_plots, print_sets
from models import train_rf_model


def main(data_dir, cone_file, cluster_file, isochrone_file, probability_threshold, candidate_filter_kwargs,
         train_columns, test_fraction):
    lp_members, members, noise, candidates, isochrone = parse_data(data_dir=data_dir,
                                                                   cone_file=cone_file,
                                                                   cluster_file=cluster_file,
                                                                   isochrone_file=isochrone_file,
                                                                   probability_threshold=probability_threshold,
                                                                   candidate_filter_kwargs=candidate_filter_kwargs)

    print('members', len(members))
    print('lp members', len(lp_members))
    print('candidates', len(candidates))
    print('noise', len(noise))

    if len(candidates) > 0:
        # create train and test datasets
        x_train, y_train, x_test, y_test, x_eval = generate_rf_input(members, noise, candidates,
                                                                     train_columns, test_fraction)

        # train random forest with high probability members and non members
        clf = train_rf_model(x_train, y_train, max_depth=10)

        # chekc how well the random forest performs on the test dataset
        score = clf.score(x_test, y_test)
        print('test set score', score)

        # predict membership of candidates
        member_candidates_indices = clf.predict(x_eval).astype(bool)

        member_candidates = candidates.iloc[member_candidates_indices]
        non_member_candidates = pd.concat((candidates, member_candidates),
                                          sort=False, ignore_index=True).drop_duplicates(keep=False)

        # print and visualize results
        print_sets(member_candidates_indices, lp_members, noise, candidates)

        make_plots(members, noise_df=noise, member_candidates_df=member_candidates,
                   non_member_candidates_df=non_member_candidates, isochrone_df=isochrone,
                   title=f'NGC_2509 (>{probability_threshold})')


if __name__ == "__main__":
    # lists of columns to use for training the random forest
    train_fields = ['parallax', 'pmra', 'pmdec', 'isochrone_d', 'ruwe']
    # train_fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    # train_fields = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp', 'ruwe']

    filter_kwargs = {'plx_sigma': 3.0,
                     'gmag_max_d': 0.8,
                     'pm_max_d': 1.0,
                     'bp_rp_max_d': 0.2}

    rf_advanced_practice_with_cone(data_dir='practice_data',
                                   cone_file='NGC_2509_cone.csv',
                                   cluster_file='NGC_2509.tsv',
                                   isochrone_file='isochrone.dat',
                                   probability_threshold=0.5,
                                   candidate_filter_kwargs=filter_kwargs,
                                   train_columns=train_fields,
                                   test_fraction=0.3)
