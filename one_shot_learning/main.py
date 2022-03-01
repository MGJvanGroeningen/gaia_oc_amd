import sys

sys.path.insert(1, '/home/mvgroeningen/git/gaia_oc_amd/')
sys.path.insert(1, '/home/matthijs/git/gaia_oc_amd/')

import torch
import os
import glob
import numpy as np
from data_filters import parse_data, get_cluster_parameters, add_train_fields, make_isochrone
from data_generation import generate_nn_input, generate_candidate_samples
from models import train_nn_model, evaluate_candidates, write_save_filename
from deepsets_zaheer import D5
from visualization import plot_sources, print_sets, make_plots
from make_cone_files import download_cone_file
from utils import save_csv, load_csv
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

    cone_path = os.path.join(data_dir, 'cones', cluster_name + '.vot.gz')
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

    if not load_from_csv and not os.path.exists(cone_path):
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
    prob_threshold = 0.3

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

    candidate_filter_kwargs = {'plx_sigma': 3.0, 'gmag_max_d': 0.75, 'pm_max_d': 1.0, 'bp_rp_max_d': 0.3}
    cluster_kwargs = get_cluster_parameters(cluster_path, cluster_name)

    print('Cluster age:', cluster_kwargs['age'])

    # Load the ischrone data and shift it in the CMD diagram using the mean parallax and extinction value
    isochrone = make_isochrone(isochrone_path, age=cluster_kwargs['age'], z=0.015, dm=cluster_kwargs['dm'])

    if load_from_csv:
        all_sources = load_csv(cluster_name, save_dir, suffix)

        hp_members = all_sources['hp_members']
        lp_members = all_sources['lp_members']
        noise = all_sources['noise']
        candidates = all_sources['candidates']
        members_compare = all_sources['compare_members']

        member_candidates = all_sources['member_candidates']
        mean_predictions = candidates['PMemb']
    else:
        all_sources = parse_data(cone_path=cone_path, members_path=members_path,
                                 compare_members_path=compare_members_path, probability_threshold=prob_threshold,
                                 isochrone=isochrone, candidate_filter_kwargs=candidate_filter_kwargs,
                                 cluster_kwargs=cluster_kwargs, max_noise=n_max_noise)

        add_train_fields(all_sources, isochrone, candidate_filter_kwargs, cluster_kwargs)

        hp_members = all_sources['hp_members']
        lp_members = all_sources['lp_members']
        noise = all_sources['noise']
        candidates = all_sources['candidates']
        members_compare = all_sources['compare_members']

    if plot:
        plot_sources(cluster_name, save_dir, hp_members, prob_threshold, candidates_df=candidates,
                     isochrone_df=isochrone, noise_df=noise, zoom=1.5, plot=plot, save=False)

    if train:
        for config in configs:
            # create training and test datasets
            training_dataset, testing_dataset = generate_nn_input(hp_members, noise, candidates, train_fields,
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
        candidate_samples = generate_candidate_samples(hp_members, noise, candidates, train_fields, n_samples,
                                                       train_on_pmemb)

        # evaluate the model on the candidate samples and return the mean probability of being a member
        mean_predictions = evaluate_candidates(candidate_samples, model)

        candidates['PMemb'] = mean_predictions

        member_candidates = candidates[candidates['PMemb'] > prob_threshold].copy()
        non_member_candidates = candidates[~candidates['source_id'].isin(member_candidates['source_id'])].copy()

        if save_subsets:
            save_csv(cluster_results_dir, hp_members, lp_members, noise, member_candidates, non_member_candidates,
                     members_compare, run_suffix=suffix)

        print(' ')
        print(f'NEW MEMBERS (>{prob_threshold}):', len(member_candidates))

        print_sets(member_candidates, lp_members, noise, candidates)

        print(' ')
        print(40 * '=')
        print(' ')

    if train or load_from_csv:
        plot_prob_threshold = 0.5
        make_plots(cluster_name, save_dir, cluster_kwargs, plot_prob_threshold, hp_members, candidates,
                   member_candidates, members_compare, noise, isochrone, mean_predictions, suffix, plot, save_plots)

    if os.path.exists(cone_path) and remove_cone:
        os.remove(cone_path)


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
        data_dir = '/data2/mvgroeningen/amd/data'
        save_dir = '/data2/mvgroeningen/amd'
        import matplotlib
        matplotlib.use('Agg')
        print('Running on strw')
    else:
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
        save_dir = os.getcwd()
        print('Running at home')

    np.random.seed(42)

    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135', 'NGC_1605']

    all_cluster_files = glob.glob(os.path.join(data_dir, 'members', '*'))
    all_clusters = sorted([os.path.basename(cluster_file).split('.')[0] for cluster_file in all_cluster_files])

    # clusters = ['NGC_752']
    clusters = all_clusters[30:50]

    multi = True
    train = True
    load_from_csv = False
    load_cp = False
    remove_log_dir = True
    plot = True
    save_subsets = True
    save_plots = True
    remove_cone = False

    if multi:
        cores = 3

        params = [(cluster, data_dir, save_dir, train, load_from_csv, load_cp, remove_log_dir, plot,
                   save_subsets, save_plots, remove_cone) for cluster in clusters]
        pool = Pool(cores)
        pool.starmap(main, params)
        pool.close()
        pool.join()
    else:
        for cluster in clusters:
            main(cluster_name=cluster, data_dir=data_dir, save_dir=save_dir, train=train, load_from_csv=load_from_csv,
                 load_cp=load_cp, remove_log_dir=remove_log_dir, plot=plot, save_subsets=save_subsets,
                 save_plots=save_plots, remove_cone=remove_cone)
