import os
import shutil

import torch
import deepsets_zaheer
import numpy as np
import torch.optim as optim
from time import sleep
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax
from sklearn.ensemble import RandomForestClassifier


def train_rf_model(x_train, y_train, max_depth=10, random_state=0):
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    return clf


def data_to_nn_input(x, y, ref):
    ref = torch.FloatTensor(ref)  # ref shape [n_support, n_features]
    x = torch.FloatTensor(x)  # x shape [n_features]
    # y = torch.LongTensor([y])  # y shape [1]
    y = torch.FloatTensor([[1 - y, y]])  # y shape [1]

    x = torch.cat((ref, x.unsqueeze(0).repeat(ref.size(0), 1)), dim=1)  # [n_support, 2 * n_features]
    x = x.reshape((1, x.size()[0], x.size()[1]))  # [1, n_support, 2 * n_features]

    return Variable(x), Variable(y)


def step(data_set, model, optimizer, criterion, l_scale, train=True):
    losses = 0
    shuffled_dataset = np.random.permutation(np.array(data_set, dtype=object))

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i, (_x, _y, ref) in enumerate(shuffled_dataset):
        x, y = data_to_nn_input(_x, _y, ref)

        if train:
            optimizer.zero_grad()

        f_x = model(x)
        loss = criterion(f_x, y)
        loss_val = loss.data.cpu().numpy() / l_scale
        losses = losses + loss_val

        if train:
            loss.backward()
            # clip_grad prevents large gradients
            # if the norm of the array containing all parameter gradients exceeds a maximum (max_norm)
            # then shrink all gradients by the same factor so that norm(gradients) = max_norm
            deepsets_zaheer.clip_grad(model, max_norm=5)
            optimizer.step()
        else:
            predicted_as_member = softmax(f_x, 1).detach().numpy()[0, 1] > 0.5
            true_member = y.detach().numpy()[0, 1] > 0.5
            if predicted_as_member and predicted_as_member == true_member:
                true_positives += 1
            elif not predicted_as_member and predicted_as_member == true_member:
                true_negatives += 1
            elif predicted_as_member and predicted_as_member != true_member:
                false_positives += 1
            else:
                false_negatives += 1
        del x, y, loss

    predictions = [true_positives, true_negatives, false_positives, false_negatives]

    return losses, predictions


def write_log_dir(logs_dir, cluster_name, config):
    model_log_dir = f'one_shot_learning_{cluster_name}_lr={str(config["lr"])}_' \
                    f'l1={str(config["l1"])}_hddnsz={str(config["hidden_size"])}_' \
                    f'prb={str(config["prob_threshold"])}_wghtim={str(config["weight_imbalance"])}_' \
                    f'xdim={str(config["x_dim"])}'
    return os.path.join(logs_dir, model_log_dir)


def write_save_filename(save_dir, cluster_name, config):
    model_log_dir = f'one_shot_learning_{cluster_name}_lr={str(config["lr"])}_' \
                    f'l1={str(config["l1"])}_hddnsz={str(config["hidden_size"])}_' \
                    f'prb={str(config["prob_threshold"])}_wghtim={str(config["weight_imbalance"])}_' \
                    f'xdim={str(config["x_dim"])}'
    return os.path.join(save_dir, model_log_dir)


def loss_scale(dataset, weight_imbalance):
    l_scale = 0
    for (_, y, _) in dataset:
        if y == 1:
            l_scale += weight_imbalance
        else:
            l_scale += 1
    return l_scale


def train_nn_model(train_dataset, test_dataset, cluster_name, save_dir, config, num_epochs=3,
                   early_stopping_threshold=5, load_checkpoint=False, remove_log_dir=False):
    lr = config['lr']
    l1 = config['l1']
    hidden_size = config['hidden_size']
    weight_imbalance = config['weight_imbalance']
    x_dim = config['x_dim']

    if weight_imbalance < 1:
        weight_class = torch.FloatTensor([1. / weight_imbalance, 1])
    else:
        weight_class = torch.FloatTensor([1, weight_imbalance])

    train_loss_scale = loss_scale(train_dataset, weight_imbalance)
    test_loss_scale = loss_scale(test_dataset, weight_imbalance)

    criterion = torch.nn.CrossEntropyLoss(weight=weight_class, reduction='sum')
    model = deepsets_zaheer.D5(hidden_size, x_dim=x_dim, pool='mean', out_dim=2)

    saved_models_dir = os.path.join(save_dir, 'saved_models')

    if not os.path.exists(saved_models_dir):
        os.mkdir(saved_models_dir)

    save_filename = write_save_filename(saved_models_dir, cluster_name, config)
    if load_checkpoint:
        if os.path.exists(save_filename):
            model.load_state_dict(torch.load(save_filename))
        else:
            print(f'Model not loaded. No model exits at {save_filename}')

    optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=l1)  # , eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, min_lr=1e-7)

    log_dir = write_log_dir(os.path.join(save_dir, 'log_dir'), cluster_name, config)

    if os.path.exists(log_dir) and remove_log_dir:
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir)

    progress_bar = tqdm(range(num_epochs),
                        unit='epochs',
                        position=1)

    if load_checkpoint:
        min_test_loss, _ = step(test_dataset, model, optimizer, criterion, test_loss_scale, train=False)
    else:
        min_test_loss = np.inf

    early_stopping_step = 0

    for e in progress_bar:
        early_stopping_step += 1

        model.train()
        train_loss, _ = step(train_dataset, model, optimizer, criterion, train_loss_scale, train=True)

        scheduler.step(train_loss)

        model.eval()
        test_loss, predictions = step(test_dataset, model, optimizer, criterion, test_loss_scale, train=False)

        print(f'\n\n{cluster_name} epoch', e)
        print('train loss', train_loss)
        print('test loss', test_loss)
        print('true_pos', predictions[0], 'true_neg', predictions[1],
              'false_pos', predictions[2], 'false_neg', predictions[3])

        sleep(1.)

        writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, e)
        writer.add_scalar('Predictions/True_positives', predictions[0], e)
        writer.add_scalar('Predictions/True_negatives', predictions[1], e)
        writer.add_scalar('Predictions/False_positives', predictions[2], e)
        writer.add_scalar('Predictions/False_negatives', predictions[3], e)

        if test_loss < min_test_loss:
            torch.save(model.state_dict(), save_filename)
            early_stopping_step = 0

        min_test_loss = min(min_test_loss, test_loss)
        if early_stopping_step >= early_stopping_threshold:
            print(f'Early stopping after {e} epochs')
            break


def evaluate_candidates(candidate_sets, model):
    n_samples = len(candidate_sets)
    n_candidates = len(candidate_sets[0])

    predictions = np.zeros((n_samples, n_candidates))
    with torch.no_grad():
        for sample_idx, candidate_set in tqdm(enumerate(candidate_sets), total=n_samples,
                                              desc="Evaluating candidate samples..."):
            for candidate_idx, (x, ref) in enumerate(candidate_set):
                x, _ = data_to_nn_input(x, 0, ref)
                member_prob = softmax(model(x), 1).detach().numpy()[0, 1]
                predictions[sample_idx, candidate_idx] = int(member_prob > 0.5)
                del x

    mean_predictions = np.mean(predictions, axis=0)

    return mean_predictions
