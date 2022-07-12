import os
import torch
import numpy as np
import torch.optim as optim
from time import sleep
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.neural_networks.deepsets_zaheer import clip_grad


def accuracies(predictions):
    """Calculates the classification accuracies based on the number of true_positives,
    true_negatives, false_positives and false_negatives.

    Args:
        predictions (list): List containing the number of true_positives, true_negatives,
            false_positives and false_negatives

    Returns:
        pos_accuracy (float): Percentage of members that were correctly classified
        neg_accuracy (float): Percentage of non-members that were correctly classified
    """
    pos_accuracy = np.round(100 * predictions[0] / (predictions[0] + predictions[3]), 1)
    neg_accuracy = np.round(100 * predictions[1] / (predictions[1] + predictions[2]), 1)
    return pos_accuracy, neg_accuracy


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
    pos_acc, neg_acc = accuracies([true_positives, true_negatives, false_positives, false_negatives])

    progress_bar.set_postfix({'loss': loss, 'pos_acc': f'{pos_acc}%', 'neg_acc': f'{neg_acc}%'})
    progress_bar.update()

    return loss, pos_acc, neg_acc


def train_model(model, train_dataset, val_dataset, model_parameters_save_path, num_epochs=40, lr=1e-6, l2=1e-5,
                weight_imbalance=1., early_stopping_threshold=5, load_model=False):
    """Trains the deep sets model on a dataset of open cluster members and non-members.

    Args:
        model (nn.Module): Deep Sets model
        train_dataset (DeepSetsDataset): Training dataset
        val_dataset (DeepSetsDataset): Validation dataset
        model_parameters_save_path (str): Path where the model parameters will be saved
            (also the path from which model parameters will be loaded)
        num_epochs (int): The number of epochs to train the model for.
        lr (float): Learning rate
        l2 (float): L2 regularization, weight decay parameter
        weight_imbalance (float): Factor by which the loss is increased when classifying members
        early_stopping_threshold (int): Number of epochs after which the training is terminated
            if the model has not improved
        load_model (bool): Whether to the (trained) model parameters from a state dict

    Returns:
        metrics (dict): Dictionary containing the loss and (non-)member classification
        accuracies for every epoch
    """
    if load_model:
        if os.path.exists(model_parameters_save_path):
            model.load_state_dict(torch.load(model_parameters_save_path))
        else:
            print(f'Model not loaded. No model exits at {model_parameters_save_path}')

    weight_class = torch.FloatTensor([1, weight_imbalance])
    criterion = torch.nn.CrossEntropyLoss(weight=weight_class, reduction='sum')
    optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=l2)

    metrics = {'train_loss': [], 'val_loss': [],
               'train_pos_acc': [], 'val_pos_acc': [],
               'train_neg_acc': [], 'val_neg_acc': []}

    loss = 0
    min_val_loss = np.inf
    early_stopping_step = 0

    for e in range(num_epochs):
        early_stopping_step += 1

        for mode, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
            loss, pos_acc, neg_acc = step(train_dataset, model, optimizer, criterion, e, mode)

            metrics[f'{mode}_loss'].append(loss)
            metrics[f'{mode}_pos_acc'].append(pos_acc)
            metrics[f'{mode}_neg_acc'].append(neg_acc)

        print(' ')
        sleep(0.1)

        # Save the model state if the loss is smaller than any previous epoch
        if loss < min_val_loss:
            torch.save(model.state_dict(), model_parameters_save_path)
            early_stopping_step = 0

        min_val_loss = min(min_val_loss, loss)

        # Stop training early if the model has not improved for 'early_stopping_threshold' epochs.
        if early_stopping_step >= early_stopping_threshold:
            print(f'Early stopping after {e} epochs')
            break

    return metrics
