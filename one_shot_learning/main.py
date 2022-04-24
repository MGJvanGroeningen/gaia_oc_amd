import sys

sys.path.insert(1, '/home/mvgroeningen/git/gaia_oc_amd/')
sys.path.insert(1, '/home/matthijs/git/gaia_oc_amd/')

import torch
import os
import glob
import numpy as np
from data_filters import parse_sources, make_isochrone, Cluster, get_cluster_parameters
from data_generation import OSLDataset, generate_osl_candidate_samples
from models import train_nn_model, evaluate_candidates, write_save_filename
from deepsets_zaheer import D5
from visualization import plot_sources, make_plots
from make_cone_files import download_cone_file
from utils import save_csv, load_csv, get_data_and_save_dir, data_dirs, save_dirs
from itertools import product
from multiprocessing import Pool
from torch.utils.data import DataLoader


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


def main(cluster, data_paths, save_paths, train=True, load_from_csv=False, load_cp=False,
         remove_log_dir=False, show=False, save_subsets=False, save_plots=False, remove_cone=False):
    if not load_from_csv and not os.path.exists(data_paths['cone']):
        download_cone_file(cluster, data_paths)

    print('Cluster:', cluster.name)

    # train params
    n_epochs = 10
    test_fraction = 0

    # model params
    hs = 32
    lr = 1e-5
    l1 = 1e-5
    weight_im = 1.
    prob_threshold = 0.8

    train_fields = ['plx_d', 'pm_d', 'bp_rp_d', 'gmag_d', 'ruwe']

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

    parameters = [hidden_sizes, lrs, l1s, weight_ims, prob_thresholds, x_dims]
    param_names = ['hidden_size', 'lr', 'l1', 'weight_imbalance', 'prob_threshold', 'x_dim']

    configs = make_hyper_param_sets(parameters, param_names)

    # data params
    n_max_members = 2000
    n_max_noise = 4000
    n_samples = 10

    candidate_filter_kwargs = {'pm_d_max': 3.0, 'plx_d_max': 3.0, 'bp_rp_d_max': 0.3, 'gmag_d_max': 1.2}

    print(vars(cluster))
    print('Cluster age:', cluster.age)

    # Load the ischrone data and shift it in the CMD diagram using the mean parallax and extinction value
    isochrone = make_isochrone(data_paths['isochrone'], age=cluster.age, dm=cluster.dm)
    # new_field_funcs, new_field_labels = train_field_functions(cluster)

    if load_from_csv:
        sources = load_csv(save_paths['cluster'])
    else:
        sources = parse_sources(data_paths['cone'], data_paths['train_members'], data_paths['comparison_members'],
                                cluster=cluster, probability_threshold=prob_threshold, isochrone=isochrone,
                                candidate_filter_kwargs=candidate_filter_kwargs)

    if show:
        plot_sources(sources, cluster, save_paths['cluster'], isochrone_df=isochrone, show_probs=False, show=show, save=False)

    if train:
        for config in configs:
            # create training and test datasets
            train_dataset = DataLoader(OSLDataset(sources, train_fields, test_fraction, n_max_members, n_max_noise,
                                                  test=False))
            test_dataset = DataLoader(OSLDataset(sources, train_fields, test_fraction, n_max_members, n_max_noise,
                                                 test=True))

            # train the model
            train_nn_model(train_dataset, test_dataset, cluster_name=cluster.name, save_paths=save_paths,
                           config=config, num_epochs=n_epochs, load_checkpoint=load_cp,
                           remove_log_dir=remove_log_dir)

        # load model
        config = configs[0]
        save_filename = write_save_filename(save_paths['model'], cluster.name, config)
        model = D5(config['hidden_size'], x_dim=x_dim, pool='mean', out_dim=2)
        model.load_state_dict(torch.load(save_filename))

        # sample candidate data with their astrometric variances and correlations
        candidate_samples = generate_osl_candidate_samples(sources, cluster, isochrone, candidate_filter_kwargs,
                                                           train_fields, n_samples)

        # evaluate the model on the candidate samples and return the mean probability of being a member
        sources.candidates['PMemb'] = evaluate_candidates(candidate_samples, model)

    if save_subsets and not load_from_csv:
        save_csv(sources, save_paths['cluster'], cluster)

    if train or load_from_csv:
        make_plots(sources, cluster, save_paths['cluster'], isochrone, show, save_plots)

    if os.path.exists(data_paths['cone']) and remove_cone:
        os.remove(data_paths['cone'])


if __name__ == "__main__":
    data_folder, save_folder = get_data_and_save_dir(model_type='one_shot_learning')
    np.random.seed(42)

    ts = 'cg18'
    # ts = 't22'
    # cs = 'cg18'
    cs = 't22'

    # all_cluster_files = glob.glob(os.path.join(data_folder, ts + '_members', '*'))
    # all_clusters = [os.path.basename(cluster_file).split('.')[0] for cluster_file in all_cluster_files]

    all_cluster_files = glob.glob(os.path.join(save_folder, 'results', '*'))
    all_clusters = [os.path.basename(cluster_file) for cluster_file in all_cluster_files]
    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135', 'NGC_1605']

    # clusters = ['NGC_1901']
    clusters = all_clusters

    multi = False
    train = False
    load_from_csv = True
    load_cp = False
    remove_log_dir = False
    show = False
    save_subsets = True
    save_plots = True
    remove_cone = False

    if multi:
        cores = 1

        params = [(cluster, data_folder, save_folder, ts, cs, train, load_from_csv, load_cp, remove_log_dir, show,
                   save_subsets, save_plots, remove_cone) for cluster in clusters]
        pool = Pool(cores)
        pool.starmap(main, params)
        pool.close()
        pool.join()
    else:
        for cluster_name in clusters:
            data_paths = data_dirs(data_folder, cluster_name, ts, cs)
            save_paths = save_dirs(save_folder, cluster_name)

            cluster_params = get_cluster_parameters(cluster_name, data_paths['cluster'])
            if cluster_params is not None:
                cluster = Cluster(cluster_params)
                main(cluster, data_paths, save_paths, train=train, load_from_csv=load_from_csv, load_cp=load_cp,
                     remove_log_dir=remove_log_dir, show=show, save_subsets=save_subsets, save_plots=save_plots,
                     remove_cone=remove_cone)
