import torch
import pandas as pd
import numpy as np
from filter_cone_data import parse_data
from data_generation import generate_nn_input, generate_candidate_samples
from models import train_nn_model, evaluate_candidates
from deepsets_zaheer import D5
from visualization import make_plots, save_csv, print_sets


if __name__ == "__main__":
    train = False
    eval_candidates = False
    load_cp = False

    plot = True
    save_plot = False
    save_filename = 'saved_models/deep_sets_model'

    np.random.seed(42)

    # make test set (split data at the start)
    # plot loss
    # figure out label indices
    # sample dataset a' = N(a_mean, a_sigma (+ covariance terms)) and use a for loop
    # use topcat to identify individual sources

    # download data based on radius (50 pc)
    # test with removing ra/dec or replace with distance to center of cluster
    # (maybe weight dimensions e.g. parallax or gmag)
    # search literature: how to deal with faint stars
    # only use astrometry for sampling

    # train params
    n_epochs = 10
    test_fraction = 0.3

    # model params
    hidden_size = 20
    weight_im = 100
    lr = 1e-4
    l1 = 1e-4

    # data params
    prob_threshold = 0.6
    n_samples = 10
    train_fields = ['parallax', 'pmra', 'pmdec', 'isochrone_d', 'ruwe']
    candidate_filter_kwargs = {'plx_sigma': 3.0,
                               'gmag_max_d': 0.8,
                               'pm_max_d': 1.0,
                               'bp_rp_max_d': 0.2}

    x_dimension = 2 * len(train_fields)
    lp_members, members, noise, candidates, isochrone = parse_data(data_dir='practice_data',
                                                                   cone_file='NGC_2509_cone_big.csv',
                                                                   cluster_file='NGC_2509.tsv',
                                                                   isochrone_file='isochrone.dat',
                                                                   probability_threshold=prob_threshold,
                                                                   candidate_filter_kwargs=candidate_filter_kwargs)
    if plot:
        make_plots(members, noise_df=noise, candidates_df=candidates, isochrone_df=isochrone,
                   title=f'NGC_2509 (>{prob_threshold})')

    print('members', len(members))
    print('lp members', len(lp_members))
    print('candidates', len(candidates))
    print('noise', len(noise))

    if train:
        # create training and test datasets
        training_dataset, testing_dataset = generate_nn_input(members, noise, candidates, train_fields, test_fraction)

        # train the model
        train_nn_model(training_dataset, testing_dataset, logs_dir='log_dir', current_lr=lr, current_l1=l1,
                       network_dim=hidden_size, x_dim=x_dimension, num_epochs=n_epochs, weight_imbalance=weight_im,
                       save_filename=save_filename, load_checkpoint=load_cp)

    if eval_candidates:
        # load model
        model = D5(hidden_size, x_dim=x_dimension, pool='mean', out_dim=2)
        model.load_state_dict(torch.load(save_filename))

        # sample candidate data with their astrometric variances and correlations
        candidate_samples = generate_candidate_samples(members, noise, candidates,
                                                       train_fields, n_samples=n_samples)

        # evaluate the model on the candidates and return the indices of the candidates that are selected as member
        member_candidates_indices = evaluate_candidates(candidate_samples, model)

        print_sets(member_candidates_indices, lp_members, noise, candidates)

        if plot:
            member_candidates = candidates.iloc[member_candidates_indices]
            non_member_candidates = pd.concat((candidates, member_candidates),
                                              sort=False, ignore_index=True).drop_duplicates(keep=False)

            make_plots(members, noise_df=noise, member_candidates_df=member_candidates,
                       non_member_candidates_df=non_member_candidates, isochrone_df=isochrone,
                       title=f'NGC_2509 (>{prob_threshold})')
            if save_plot:
                save_csv(members, noise, member_candidates, non_member_candidates,)
