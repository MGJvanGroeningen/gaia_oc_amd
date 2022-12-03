import os
import torch
import numpy as np
import torch.optim as optim
from time import sleep
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.machine_learning.deepsets_zaheer import clip_grad
from gaia_oc_amd.io import save_hyper_parameters, save_model


def calculate_metrics(tp, tn, fp, fn):
    """Calculates a number of metrics based on the number of true positives,
    true negatives, false positives and false negatives.

    Args:
        tp (int): Number of true positives
        tn (int): Number of true negatives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        precision (float): Fraction of positive classifications that were correct
        recall (float): Fraction of positive examples that were correctly classified (true positive rate)
        selectivity (float): Fraction of negative examples that were correctly classified (true negative rate)
        accuracy (float): Fraction of examples that were correctly classified
        balanced_accuracy (float): Mean of the recall and selectivity
        f1 (float): F1 score, harmonic mean of the precision and recall
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    selectivity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall + selectivity) / 2
    f1 = 2 * tp / (2 * tp + fp + fn)

    metric_dict = {'precision': precision,
                   'recall': recall,
                   'selectivity': selectivity,
                   'accuracy': accuracy,
                   'balanced_accuracy': balanced_accuracy,
                   'f1': f1}

    return metric_dict


def step(data_set, model, optimizer, criterion, epoch, mode='train'):
    """Trains the deep sets model on a dataset of open cluster members and non-members.

    Args:
        data_set (DeepSetsDataset): Deep Sets dataset
        model (nn.Module): Deep Sets model
        optimizer (Optimizer): Optimizer for updating the model parameters
        criterion (nn.CrossEntropyLoss): Loss function
        epoch (int): The current epoch
        mode (str): Training or validation/test mode, anything other than 'train'
            is considered as validation/test mode

    Returns:
        loss (float): The mean loss of the epoch
        pos_acc (float): Percentage of members that were correctly classified
        neg_acc (float): Percentage of non-members that were correctly classified
    """
    losses = []
    data_size = len(data_set)
    progress_bar = tqdm(np.arange(data_size), desc=f'{mode} epoch {epoch}', total=data_size)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    if mode == 'train':
        model.train()
    else:
        model.eval()

    for i, (x, y) in enumerate(data_set):
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())

        predictions = softmax(out, 1).detach().numpy()[:, 1] > 0.5
        truths = y.detach().numpy()[:, 1] > 0.5
        output_pairs = np.stack((predictions, truths), axis=-1)

        true_positives += np.sum(np.all(output_pairs == np.array([True, True]), axis=1))
        true_negatives += np.sum(np.all(output_pairs == np.array([False, False]), axis=1))
        false_positives += np.sum(np.all(output_pairs == np.array([True, False]), axis=1))
        false_negatives += np.sum(np.all(output_pairs == np.array([False, True]), axis=1))

        if mode == 'train':
            loss.backward()
            # clip_grad prevents large gradients
            # if the norm of the array containing all parameter gradients exceeds a maximum (max_norm)
            # then shrink all gradients by the same factor so that norm(gradients) = max_norm
            clip_grad(model, max_norm=5)
            optimizer.step()

            del x, y, loss
            optimizer.zero_grad()
        else:
            del x, y, loss
        if i < len(data_set) - 1:
            progress_bar.update()

    loss = np.mean(losses)
    metrics = calculate_metrics(true_positives, true_negatives, false_positives, false_negatives)

    progress_bar.set_postfix({'loss': loss, 'f1': f'{np.round(100 * metrics["f1"], 1)}%'})
    progress_bar.update()

    return loss, metrics


def train_model(model, model_save_dir, train_dataset, val_dataset=None, num_epochs=40, lr=1e-6, l2=1e-5,
                weight_imbalance=1., early_stopping_threshold=10, load_model=False):
    """Trains the deep sets model on a dataset of open cluster members and non-members.

    Args:
        model (nn.Module): Deep Sets model
        model_save_dir (str): Path to the directory where the model parameters will be saved
        train_dataset (DeepSetsDataset): Training dataset
        val_dataset (DeepSetsDataset): Validation dataset
        num_epochs (int): The number of epochs to train the model for.
        lr (float): Learning rate
        l2 (float): L2 regularization, weight decay parameter
        weight_imbalance (float): Factor by which the loss is increased when classifying members
        early_stopping_threshold (int): Number of epochs after which the training is terminated
            if the validation F1-score has not improved
        load_model (bool): Whether to load model parameters from a state dict

    Returns:
        metrics (dict): Dictionary containing the loss and (non-)member classification
        accuracies for every epoch
    """
    model_parameters_path = os.path.join(model_save_dir, 'model_parameters')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    else:
        if load_model:
            if os.path.exists(model_parameters_path):
                model.load_state_dict(torch.load(model_parameters_path))
            else:
                print(f'Model not loaded. No model exists at {model_parameters_path}')

    training_clusters = train_dataset.dataset.cluster_names
    if val_dataset is not None:
        validation_clusters = val_dataset.dataset.cluster_names
    else:
        validation_clusters = []

    hyperparameters = {'n_training_clusters': len(training_clusters),
                       'training_clusters': sorted(training_clusters),
                       'n_validation_clusters': len(validation_clusters),
                       'validation_clusters': sorted(validation_clusters),
                       'validation_fraction': len(validation_clusters) / (len(training_clusters) +
                                                                          len(validation_clusters)),
                       'size_support_set': train_dataset.dataset.size_support_set,
                       'batch_size': train_dataset.batch_size,
                       'n_pos_duplicates': train_dataset.dataset.n_pos_duplicates,
                       'neg_pos_ratio': train_dataset.dataset.neg_pos_ratio,
                       'source_features': train_dataset.dataset.source_feature_names,
                       'source_feature_means': list(train_dataset.dataset.source_feature_means),
                       'source_feature_stds': list(train_dataset.dataset.source_feature_stds),
                       'cluster_features': train_dataset.dataset.cluster_feature_names,
                       'cluster_feature_means': list(train_dataset.dataset.cluster_feature_means),
                       'cluster_feature_stds': list(train_dataset.dataset.cluster_feature_stds),
                       'hidden_size': model.d_dim,
                       'n_epochs': num_epochs,
                       'lr': lr,
                       'l2': l2,
                       'weight_imbalance': weight_imbalance,
                       'early_stopping_threshold': early_stopping_threshold,
                       'seed': train_dataset.dataset.seed}

    save_hyper_parameters(model_save_dir, hyperparameters)

    weight_class = torch.FloatTensor([1, weight_imbalance])
    criterion = torch.nn.CrossEntropyLoss(weight=weight_class, reduction='sum')
    optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=l2)

    metrics_dict = {'train_loss': [], 'val_loss': [],
                    'train_precision': [], 'val_precision': [],
                    'train_recall': [], 'val_recall': [],
                    'train_selectivity': [], 'val_selectivity': [],
                    'train_accuracy': [], 'val_accuracy': [],
                    'train_balanced_accuracy': [], 'val_balanced_accuracy': [],
                    'train_f1': [], 'val_f1': []}

    f1 = 0
    max_f1 = 0
    epochs_without_improvement = 0

    for e in range(num_epochs):
        for mode, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
            if dataset is not None:
                loss, metrics = step(dataset, model, optimizer, criterion, e, mode)

                metrics_dict[f'{mode}_loss'].append(loss)
                metrics_dict[f'{mode}_precision'].append(metrics['precision'])
                metrics_dict[f'{mode}_recall'].append(metrics['recall'])
                metrics_dict[f'{mode}_selectivity'].append(metrics['selectivity'])
                metrics_dict[f'{mode}_accuracy'].append(metrics['accuracy'])
                metrics_dict[f'{mode}_balanced_accuracy'].append(metrics['balanced_accuracy'])
                metrics_dict[f'{mode}_f1'].append(metrics['f1'])

                # If a validation set provided, keep track of its F1 score, otherwise
                # keep track of the training set F1 score
                if not (mode == 'train' and val_dataset is not None):
                    f1 = metrics['f1']

        # Prevent printing of progress bars from messing up
        print(' ')
        sleep(0.1)

        # Save the model state if the F1 score is better than any previous epoch
        if f1 > max_f1:
            save_model(model_save_dir, model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        max_f1 = max(max_f1, f1)

        # Stop training early if the model has not improved for 'early_stopping_threshold' epochs.
        if epochs_without_improvement >= early_stopping_threshold:
            print(f'Early stopping after {e} epochs')
            break

    return metrics_dict
