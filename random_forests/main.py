import os
import glob
import numpy as np
from data_filters import parse_data, Cluster, make_isochrone, isochrone_delta, train_field_functions
from data_generation import generate_rf_input, generate_rf_candidate_samples
from visualization import make_plots
from models import train_rf_model
from utils import get_data_and_save_dir, data_dirs, results_dirs


def main(cluster_name, data_dir, save_dir, train_set, comparison_set, train_columns, test_fraction):
    _ = results_dirs(save_dir, cluster_name)
    data_paths = data_dirs(data_dir, cluster_name, train_set, comparison_set)

    prob_threshold = 0.8
    n_samples = 10

    cluster = Cluster(cluster_name, data_paths['cluster'])
    candidate_filter_kwargs = {'pm_d_max': 3.0, 'plx_d_max': 3.0, 'isochrone_d_max': 1.2, 'bp_rp_d_max': 0.3,
                               'gmag_d_max': 1.0}

    print(vars(cluster))
    print('Cluster age:', cluster.age)

    # Load the ischrone data and shift it in the CMD diagram using the mean parallax and extinction value
    isochrone = make_isochrone(data_paths['isochrone'], age=cluster.age, dm=cluster.dm)

    sources = parse_data(data_paths['cone'], data_paths['train_members'], data_paths['comparison_members'],
                         cluster=cluster, probability_threshold=prob_threshold, isochrone=isochrone,
                         candidate_filter_kwargs=candidate_filter_kwargs)

    new_field_funcs, new_field_labels = train_field_functions(cluster)
    new_field_funcs.append(isochrone_delta(isochrone, candidate_filter_kwargs))
    new_field_labels.append(['bp_rp_d', 'gmag_d'])
    for f, label in zip(new_field_funcs, new_field_labels):
        sources.add_field(f, label)

    # create train and test datasets
    x_train, y_train, x_test, y_test = generate_rf_input(sources, train_columns, test_fraction)

    # train random forest with high probability members and non members
    clf = train_rf_model(x_train, y_train, max_depth=10)

    # chekc how well the random forest performs on the test dataset
    score = clf.score(x_test, y_test)
    print('test set score', score)

    candidate_samples = generate_rf_candidate_samples(sources, train_columns, new_field_funcs[0], new_field_funcs[1],
                                                      new_field_funcs[-1], n_samples)

    predictions = []
    for i in range(n_samples):
        pred = clf.predict(candidate_samples[i]).astype(int)
        predictions.append(pred)

    sources.candidates['PMemb'] = np.mean(predictions)
    make_plots(sources, cluster, save_dir, isochrone, show=True, save_plots=True)


if __name__ == "__main__":
    data_dir, save_dir = get_data_and_save_dir(model_type='random_forests')
    np.random.seed(42)

    all_cluster_files = glob.glob(os.path.join(data_dir, 'members', '*'))
    all_clusters = sorted([os.path.basename(cluster_file).split('.')[0] for cluster_file in all_cluster_files])

    # clusters = ['NGC_752', 'NGC_2509', 'Collinder_394', 'Ruprecht_33', 'IC_2714', 'Ruprecht_135', 'NGC_1605']
    # cluster = all_clusters[30:50]
    clusters = ['NGC_1901']

    train_col = ['plx_d', 'pm_d', 'bp_rp_d', 'gmag_d', 'ruwe']

    # ts = 'cg18'
    ts = 't22'
    # cs = 'cg18'
    cs = 't22'

    for cluster_name in clusters:
        main(cluster_name=cluster_name, data_dir=data_dir, save_dir=save_dir, train_set=ts, comparison_set=cs,
             train_columns=train_col, test_fraction=0.3)
