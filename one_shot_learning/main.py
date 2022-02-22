import sys

sys.path.insert(1, '/home/mvgroeningen/git/gaia_oc_amd/')

import torch
import os
import glob
import shutil
import pandas as pd
import numpy as np
from filter_cone_data import parse_data, get_cluster_parameters, parse_members, add_train_fields, number_of_chunks, \
    make_isochrone
from data_generation import generate_nn_input, generate_candidate_samples
from models import train_nn_model, evaluate_candidates, write_save_filename
from deepsets_zaheer import D5
from visualization import make_plots, save_csv, print_sets, plot_density_profile, plot_mass_segregation_profile, \
    compare_members, load_csv, projected_coordinates
from make_cone_files import download_cone_file
from itertools import product
from multiprocessing import Pool


def make_hyper_param_sets(parameters, parameter_names):
    assert len(parameters) == len(parameter_names)

    parameter_combinations = product(*parameters)

    hyper_parameter_sets = []

    for combination in parameter_combinations:
        hyper_param_set = {}
        for parameter, name in zip(combination, parameter_names):
            hyper_param_set.update({name: parameter})
        hyper_parameter_sets.append(hyper_param_set)

    return hyper_parameter_sets


def main(cluster_name, data_dir, save_dir, train=True, load_from_csv=False, load_cp=False, remove_log_dir=False,
         plot=False, save_subsets=False, save_plots=False, remove_cone=False):
    suffix = None

    cone_path = os.path.join(data_dir, 'cones', cluster_name + '.csv')
    members_path = os.path.join(data_dir, 'members', cluster_name + '.csv')
    compare_members_path = os.path.join(data_dir, 'compare_members', cluster_name + '.csv')
    isochrone_path = os.path.join(data_dir, 'isochrones', 'isochrones.dat')
    cluster_path = os.path.join(data_dir, 'cluster_parameters.tsv')

    results_dir = os.path.join(save_dir, 'results')
    saved_models_dir = os.path.join(save_dir, 'saved_models')
    cluster_results_dir = os.path.join(results_dir, cluster_name)

    for directory in [results_dir, saved_models_dir, cluster_results_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    if not os.path.exists(cone_path):
        download_cone_file(cluster_name, data_dir)

    print('Cluster:', cluster_name)

    if os.path.exists(compare_members_path):
        print('Comparison cluster exists')
    else:
        print('No comparison cluster exists')

    # train params
    n_epochs = 30
    test_fraction = 0.33

    # model params
    hs = 10
    lr = 4e-5
    l1 = 4e-4
    weight_im = 1.
    prob_threshold = 0.7

    train_fields = ['plx_d', 'pm_d', 'bp_rp_d', 'gmag_d', 'ruwe']
    # train_fields = ['parallax', 'pmra', 'pmdec', 'pm_d', 'bp_rp_d', 'gmag_d', 'ruwe']

    train_on_pmemb = False
    if train_on_pmemb:
        x_dim = 2 * (len(train_fields) + 1)
    else:
        x_dim = 2 * len(train_fields)

    # hidden_sizes = [15, 20, 25]
    # lrs = [1e-4, 2e-4, 4e-4]
    # l1s = [1e-4, 2e-4, 4e-4]
    # weight_ims = [1., 2., 5.]
    # prob_thresholds = [0.6, 0.7, 0.8]

    hidden_sizes = [hs]
    lrs = [lr]
    l1s = [l1]
    weight_ims = [weight_im]
    prob_thresholds = [prob_threshold]
    x_dims = [x_dim]

    params = [hidden_sizes, lrs, l1s, weight_ims, prob_thresholds, x_dims]
    param_names = ['hidden_size', 'lr', 'l1', 'weight_imbalance', 'prob_threshold', 'x_dim']

    configs = make_hyper_param_sets(params, param_names)

    # data params
    n_max_members = 2000
    n_max_noise = 10000
    n_samples = 20
    chunk_size = 1000000

    candidate_filter_kwargs = {'plx_sigma': 3.0, 'gmag_max_d': 0.75, 'pm_max_d': 1.0, 'bp_rp_max_d': 0.3}
    cluster_kwargs = get_cluster_parameters(cluster_path, cluster_name)

    print('Cluster age:', cluster_kwargs['age'])

    # Load the ischrone data and shift it in the CMD diagram using the mean parallax and extinction value
    isochrone = make_isochrone(isochrone_path, age=cluster_kwargs['age'], z=0.015, dm=cluster_kwargs['dm'],
                               extinction_v=cluster_kwargs['a_v'])

    if train or not load_from_csv:
        all_sources = parse_data(cone_path=cone_path, members_path=members_path,
                                 probability_threshold=prob_threshold, isochrone=isochrone,
                                 candidate_filter_kwargs=candidate_filter_kwargs, cluster_kwargs=cluster_kwargs,
                                 chunk_size=chunk_size, max_noise=n_max_noise)

        add_train_fields(all_sources, isochrone, candidate_filter_kwargs, cluster_kwargs)

        members = all_sources['hp_members']
        lp_members = all_sources['lp_members']
        noise = all_sources['noise']
        candidates = all_sources['candidates']

        if train:
            if plot:
                make_plots(cluster_name, save_dir, members, prob_threshold, candidates_df=candidates,
                           isochrone_df=isochrone, noise_df=noise, zoom=1.5, plot=plot, save=False)

            for config in configs:
                # create training and test datasets
                training_dataset, testing_dataset = generate_nn_input(members, noise, candidates, train_fields,
                                                                      test_fraction, n_max_members, train_on_pmemb)

                # train the model
                train_nn_model(training_dataset, testing_dataset, cluster_name=cluster_name, save_dir=save_dir,
                               config=config, num_epochs=n_epochs, load_checkpoint=load_cp,
                               remove_log_dir=remove_log_dir)

        config = configs[0]

        # load model
        save_filename = write_save_filename(saved_models_dir, cluster_name, config)

        model = D5(config['hidden_size'], x_dim=x_dim, pool='mean', out_dim=2)
        model.load_state_dict(torch.load(save_filename))

        # sample candidate data with their astrometric variances and correlations
        candidate_samples = generate_candidate_samples(members, noise, candidates, train_fields, n_samples,
                                                       train_on_pmemb)

        # evaluate the model on the candidate samples and return the mean probability of being a member
        mean_predictions = evaluate_candidates(candidate_samples, model)

        candidates['PMemb'] = mean_predictions

        member_candidates = candidates[candidates['PMemb'] > prob_threshold].copy()
        non_member_candidates = candidates[~candidates['source_id'].isin(member_candidates['source_id'])].copy()

        print(' ')
        print(f'NEW MEMBERS (>{prob_threshold}):', len(member_candidates))

        print_sets(member_candidates, lp_members, noise, candidates)

        print(' ')
        print(40 * '=')
        print(' ')
    else:
        members, noise, candidates, member_candidates, non_member_candidates = load_csv(cluster_name, save_dir, suffix)
        mean_predictions = candidates['PMemb']

    if save_subsets and (train or not load_from_csv):
        save_csv(cluster_results_dir, members, noise, member_candidates, non_member_candidates, run_suffix=suffix)
        if os.path.exists(cone_path) and remove_cone:
            os.remove(cone_path)


    plot_prob_threshold = 0.5
    old_members = members[members['PMemb'] >= plot_prob_threshold].copy()
    new_members = pd.concat((old_members, member_candidates), sort=False, ignore_index=True)

    r_t = plot_density_profile(cluster_name, save_dir, [members, new_members], plot_prob_threshold, cluster_kwargs,
                               run_suffix=suffix, plot=plot, save=save_plots)

    x, y = projected_coordinates(member_candidates, cluster_kwargs)
    member_candidates['r'] = np.sqrt(x ** 2 + y ** 2)
    member_candidates = member_candidates[(member_candidates['r'] <= r_t) &
                                          (member_candidates['PMemb'] >= plot_prob_threshold)].copy()
    new_members = pd.concat((old_members, member_candidates), sort=False, ignore_index=True)

    plot_mass_segregation_profile(cluster_name, save_dir, [members, new_members], plot_prob_threshold, cluster_kwargs,
                                  run_suffix=suffix, plot=plot, save=save_plots)

    make_plots(cluster_name, save_dir, members, plot_prob_threshold, candidates_df=candidates, noise_df=noise,
               isochrone_df=isochrone, mean_predictions=mean_predictions, zoom=1.5, dot_size=5.0, cmap='autumn_r',
               run_suffix=suffix, plot=plot, save=save_plots)
    make_plots(cluster_name, save_dir, members, plot_prob_threshold, member_candidates_df=member_candidates,
               mean_predictions=mean_predictions, isochrone_df=isochrone, zoom=1.0, dot_size=5.0, cmap='autumn_r',
               run_suffix=suffix, plot=plot, save=save_plots)

    if os.path.exists(compare_members_path):
        n_chunks = number_of_chunks(cone_path, chunk_size)

        members_to_compare_to, _ = parse_members(cone_path, compare_members_path, plot_prob_threshold,
                                                 cluster_kwargs, n_chunks, chunk_size)

        compare_members(cluster_name, save_dir, old_members, member_candidates, members_to_compare_to,
                        plot_prob_threshold, x_fields=['ra', 'bp_rp', 'pmra', 'phot_g_mean_mag'],
                        y_fields=['dec', 'phot_g_mean_mag', 'pmdec', 'parallax'], run_suffix=suffix, plot=plot,
                        save=save_plots)


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

# output of softmax not probability? revert to >0.5 -> 1 and <0.5 -> 0
# implement early stopping
# implement multiprocessing
# check out mass segregation paper on how to present mass segregation results
# try to runs on multiple computers

if __name__ == "__main__":
    if os.getcwd().split('/')[2] == 'mvgroeningen':
        data_dir = '/data1/mvgroeningen/amd/data'
        save_dir = '/data1/mvgroeningen/amd'
        print('Running on strw')
    else:
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
        save_dir = os.getcwd()
        print('Running at home')

    np.random.seed(42)

    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135', 'NGC_1605']

    # clusters = ['NGC_1605']
    all_cluster_files = glob.glob(os.path.join(data_dir, 'members', '*'))
    all_clusters = sorted([os.path.basename(cluster_file).split('.')[0] for cluster_file in all_cluster_files])
    clusters = all_clusters[4:30]

    train = True
    load_from_csv = False
    load_cp = False
    remove_log_dir = False
    plot = False
    save_subsets = True
    save_plots = True
    remove_cone = True

    cores = 1

    params = [(cluster, data_dir, save_dir, train, load_from_csv, load_cp, remove_log_dir, plot,
               save_subsets, save_plots, remove_cone) for cluster in clusters]
    pool = Pool(cores)
    pool.starmap(main, params)
    pool.close()
    pool.join()
