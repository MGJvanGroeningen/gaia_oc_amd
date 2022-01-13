import os
import tqdm
import torch
import deepsets_zaheer
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax
from sklearn.ensemble import RandomForestClassifier


def train_rf_model(x_train, y_train, max_depth=10, random_state=0):
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    return clf


def data_to_nn_input(data_point):
    # x shape [n_features]
    # y shape [1]
    # ref shape [n_support, n_features]

    x, y, ref = data_point

    ref = torch.FloatTensor(ref)
    x = torch.FloatTensor(x)
    y = torch.LongTensor([y])

    x = torch.cat((ref, x.unsqueeze(0).repeat(ref.size(0), 1)), dim=1)  # [n_support, 2 * n_features]
    x = x.reshape((1, x.size()[0], x.size()[1]))
    x = Variable(x)  # [1, n_support, 2 * n_features]
    y = Variable(y)  # [1]
    return x, y


def step(data_set, model, optimizer, criterion, train=True):
    losses = 0
    n_data = len(data_set)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i, data_point in enumerate(np.random.permutation(data_set)):
        x, y = data_to_nn_input(data_point)

        if train:
            optimizer.zero_grad()

        f_x = model(x)
        loss = criterion(f_x, y)
        loss_val = loss.data.cpu().numpy()
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
            true_member = bool(y.detach().numpy()[0])
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

    return losses / n_data, predictions


def write_log_dir(logs_dir, config):
    config_dir = f'test_run_lr={str(config["current_lr"])}_' \
                 f'l1={str(config["current_l1"])}_netdim={str(config["network_dim"])}'
    return os.path.join(logs_dir, config_dir)


def train_nn_model(train_dataset, test_dataset, logs_dir, current_lr=1e-4, current_l1=1e-4, network_dim=10,
                   x_dim=18, num_epochs=3, weight_imbalance=100, save_filename=None, load_checkpoint=False):
    if weight_imbalance < 1:
        weight_class = torch.FloatTensor([1. / weight_imbalance, 1])
    else:
        weight_class = torch.FloatTensor([1, weight_imbalance])

    criterion = torch.nn.CrossEntropyLoss(weight=weight_class, reduction='sum')
    model = deepsets_zaheer.D5(network_dim, x_dim=x_dim, pool='mean', out_dim=2)
    if load_checkpoint:
        if os.path.exists(save_filename):
            model.load_state_dict(torch.load(save_filename))
        else:
            print(f'Model not loaded. No model exits at {save_filename}')
    optimizer = optim.Adam([{'params': model.parameters()}], lr=current_lr, weight_decay=current_l1)  # , eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, min_lr=1e-7)
    config = {'current_lr': current_lr,
              'current_l1': current_l1,
              'network_dim': network_dim}

    log_dir = write_log_dir(logs_dir, config)
    writer = SummaryWriter(log_dir)

    progress_bar = tqdm.tqdm(range(num_epochs),
                             unit='epochs',
                             position=0)

    for e in progress_bar:
        model.train()
        train_losses, _ = step(train_dataset, model, optimizer, criterion, train=True)

        scheduler.step(train_losses)
        print('epoch', e, 'train loss', train_losses)

        model.eval()
        test_losses, predictions = step(test_dataset, model, optimizer, criterion, train=False)

        print('epoch', e, 'test loss', test_losses)
        print('true_pos', predictions[0], 'true_neg', predictions[1],
              'false_pos', predictions[2], 'false_neg', predictions[3])

        writer.add_scalars('Loss', {'train': train_losses, 'test': test_losses}, e)
        writer.add_scalar('Predictions/True_positives', predictions[0], e)
        writer.add_scalar('Predictions/True_negatives', predictions[1], e)
        writer.add_scalar('Predictions/False_positives', predictions[2], e)
        writer.add_scalar('Predictions/False_negatives', predictions[3], e)

    if save_filename is not None:
        torch.save(model.state_dict(), save_filename)


def evaluate_candidates(candidate_sets, model):
    n_samples = len(candidate_sets)
    n_candidates = len(candidate_sets[0])

    predictions = np.zeros((n_samples, n_candidates))
    with torch.no_grad():
        for sample_idx, candidate_set in enumerate(candidate_sets):
            for candidate_idx, (x, ref) in enumerate(candidate_set):
                x, _ = data_to_nn_input((x, 0, ref))
                member_prob = softmax(model(x), 1).detach().numpy()[0, 1]
                predictions[sample_idx, candidate_idx] = member_prob
                del x

    member_indices = np.where(np.mean(predictions, axis=0) > 0.5)[0]

    return member_indices
