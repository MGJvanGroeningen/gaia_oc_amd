import os
import argparse
from torch.utils.data import DataLoader

from gaia_oc_amd.data_preparation.datasets import multi_cluster_deep_sets_datasets, global_feature_mean_and_std, \
    DeepSetsDataset
from gaia_oc_amd.data_preparation.io import save_hyper_parameters, cluster_list

from gaia_oc_amd.neural_networks.training import train_model
from gaia_oc_amd.neural_networks.deepsets_zaheer import D5

from gaia_oc_amd.candidate_evaluation.visualization import plot_loss_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')
    parser.add_argument('--model_save_dir', nargs='?', type=str, default='deep_sets_model',
                        help='Path to where the model will be saved.')
    parser.add_argument('--validation_fraction', nargs='?', type=float, default=0.3,
                        help='Fraction of the data to use for validation.')
    parser.add_argument('--size_support_set', nargs='?', type=int, default=5,
                        help='The number of members in the support set.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch size of the training and validation datasets.')
    parser.add_argument('--max_members', nargs='?', type=int, default=100,
                        help='Maximum number of members used per cluster.')
    parser.add_argument('--max_non_members', nargs='?', type=int, default=1000,
                        help='Maximum number of non-members used per cluster.')
    parser.add_argument('--training_features', nargs='*', type=str, default=['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g'],
                        help='Features on which the model will be trained.')
    parser.add_argument('--hidden_size', nargs='?', type=int, default=64,
                        help='Hidden size of the neural network layers.')
    parser.add_argument('--load_model', nargs='?', type=bool, default=False,
                        help='Whether to load the parameters of an already trained model, saved at "save_path".')
    parser.add_argument('--n_epochs', nargs='?', type=int, default=40,
                        help='The number of training epochs.')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--l2', nargs='?', type=float, default=1e-5,
                        help='L2 regularization.')
    parser.add_argument('--weight_imbalance', nargs='?', type=float, default=2.,
                        help='Weight imbalance. '
                             'A higher value gives more weight to classifying members as opposed to non-members.')
    parser.add_argument('--early_stopping_threshold', nargs='?', type=int, default=5,
                        help='Number of epochs after which the training is terminated if the model has not improved.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the distribution of train and validation data.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the loss and accuracy plot.')
    parser.add_argument('--save_plot', nargs='?', type=bool, default=True,
                        help='Whether to save the loss and accuracy plot.')

    args_dict = vars(parser.parse_args())

    # main arguments
    data_dir = args_dict['data_dir']
    cluster_names = cluster_list(args_dict['cluster_names'], data_dir)

    # path arguments
    model_save_dir = args_dict['model_save_dir']

    # data arguments
    validation_fraction = args_dict['validation_fraction']
    size_support_set = args_dict['size_support_set']
    batch_size = args_dict['batch_size']
    max_members = args_dict['max_members']
    max_non_members = args_dict['max_non_members']
    training_features = args_dict['training_features']

    # model arguments
    hidden_size = args_dict['hidden_size']
    load_model = args_dict['load_model']

    # training arguments
    n_epochs = args_dict['n_epochs']
    lr = args_dict['lr']
    l2 = args_dict['l2']
    weight_imbalance = args_dict['weight_imbalance']
    early_stopping_threshold = args_dict['early_stopping_threshold']
    seed = args_dict['seed']

    # plot arguments
    show = args_dict['show']

    # save arguments
    save_plot = args_dict['save_plot']

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # Create the model
    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)

    # Store feature means and standard deviations for future normalization during candidate evaluation
    training_feature_means, training_feature_stds = global_feature_mean_and_std(data_dir, cluster_names,
                                                                                training_features)

    hyperparameters = {'cluster_names': cluster_names,
                       'training_feature_means': list(training_feature_means),
                       'training_feature_stds': list(training_feature_stds),
                       'validation_fraction': validation_fraction,
                       'size_support_set': size_support_set,
                       'batch_size': batch_size,
                       'max_members': max_members,
                       'max_non_members': max_non_members,
                       'training_features': training_features,
                       'hidden_size': hidden_size,
                       'n_epochs': n_epochs,
                       'lr': lr,
                       'l2': l2,
                       'weight_imbalance': weight_imbalance,
                       'seed': seed}

    save_hyper_parameters(model_save_dir, hyperparameters)

    # Create training and validation datasets
    train_dataset, val_dataset = multi_cluster_deep_sets_datasets(data_dir, cluster_names, training_features,
                                                                  training_feature_means, training_feature_stds,
                                                                  validation_fraction=validation_fraction,
                                                                  max_members=max_members,
                                                                  max_non_members=max_non_members,
                                                                  size_support_set=size_support_set, seed=seed)

    train_dataset = DataLoader(DeepSetsDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(DeepSetsDataset(val_dataset), batch_size=batch_size, shuffle=True)

    # Train the model
    print(' ')
    model_parameters_save_path = os.path.join(model_save_dir, 'model_parameters')
    metrics = train_model(model, train_dataset, val_dataset, model_parameters_save_path=model_parameters_save_path,
                          num_epochs=n_epochs, lr=lr, l2=l2, weight_imbalance=weight_imbalance,
                          early_stopping_threshold=early_stopping_threshold, load_model=load_model)
    print(f'Saved model parameters at {os.path.abspath(model_parameters_save_path)}')

    # Show training progress
    plot_loss_accuracy(metrics, model_save_dir, show, save_plot)
    if save_plot:
        print(f'Created loss and accuracy plot at {os.path.abspath(os.path.join(model_save_dir, "loss_accuracy.png"))}')
