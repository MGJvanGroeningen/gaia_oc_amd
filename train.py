import numpy as np
from filter_cone_data import generate_training_data, generate_candidate_data
import deepsets_zaheer
import torch
import torch.optim as optim
from torch.autograd import Variable


def train_nn(train_dataset,
             current_lr=1e-4,
             current_l1=1e-4,
             network_dim=10,
             x_dim=18,
             num_epochs=3,
             save=None):

    criterion = torch.nn.CrossEntropyLoss()
    model = deepsets_zaheer.D5(network_dim, x_dim=x_dim, pool='mean', out_dim=2)
    optimizer = optim.Adam([{'params': model.parameters()}], lr=current_lr, weight_decay=current_l1)  # , eps=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, min_lr=1e-7)

    for e in range(num_epochs):
        losses = 0
        model.train()

        print(len(train_dataset))

        for i, (x, y, ref) in enumerate(np.random.permutation(train_dataset)):

            # x shape [n_features]
            # y shape [1]
            # ref shape [n_support, n_features]

            ref = torch.FloatTensor(ref)
            x = torch.FloatTensor(x)
            y = torch.LongTensor([y])

            X = torch.cat((ref, x.unsqueeze(0).repeat(ref.size(0), 1)), dim=1)  # [n_support, 2 * n_features]
            X = X.reshape((1, X.size()[0], X.size()[1]))
            X = Variable(X)  # [1, n_support, 2 * n_features]
            Y = Variable(y)  # [1]

            optimizer.zero_grad()
            prediction = model(X)  # [1, 2]

            loss = criterion(prediction, Y)
            loss_val = loss.data.cpu().numpy()
            losses = losses + loss_val
            loss.backward()

            # clip_grad prevents large gradients
            # if the norm of the array containing all parameter gradients exceeds a maximum (max_norm)
            # then shrink all gradients by the same factor so that norm(gradients) = max_norm
            deepsets_zaheer.clip_grad(model, max_norm=5)

            optimizer.step()
            del X, Y, prediction, loss

        scheduler.step(losses)
        print('epoch', e, 'loss', losses)

    if save is not None:
        torch.save(model.state_dict(), save)


def evaluate_candidates(candidates, model):
        predictions = []
        with torch.no_grad():
            for (x, ref) in candidates:

                ref = torch.FloatTensor(ref)
                x = torch.FloatTensor(x)

                X = torch.cat((ref, x.unsqueeze(0).repeat(ref.size(0), 1)), dim=1)
                X = X.reshape((1, X.size()[0], X.size()[1]))
                X = Variable(X)  # [1, n_support, 2 * n_features]

                print(model(X))

                prediction = torch.nn.functional.log_softmax(model(X), 1).detach().numpy()
                # print(prediction)
                predictions.append(prediction > 0.5)
                del X

        print('Number of candidates:', len(predictions))
        print('Number of candidates identified as member:', np.sum(np.array(predictions)))



if __name__ == "__main__":
    train = True
    hidden_size = 10
    x_dimension = 18

    # make test set (split data at the start)
    # plot loss
    # figure out label indices

    save_filename = 'models/deep_sets_model'
    candidate_select_columns = ['ra', 'DEC', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp',
                                'ruwe']

    if train:
        training_dataset = generate_training_data(data_dir='practice_data',
                                                  cone_file='NGC_2509_cone.csv',
                                                  cluster_file='NGC_2509.tsv',
                                                  candidate_selection_columns=candidate_select_columns,
                                                  probability_threshold=0.9,
                                                  plx_sigma=3.0,
                                                  gmag_max_d=0.2,
                                                  pm_max_d=0.5,
                                                  bp_rp_max_d=0.05)

        train_nn(training_dataset,
                 current_lr=1e-4,
                 current_l1=1e-4,
                 network_dim=10,
                 x_dim=9 * 2,
                 num_epochs=30,
                 save=save_filename)
    else:
        model = deepsets_zaheer.D5(hidden_size, x_dim=x_dimension, pool='mean', out_dim=2)
        model.load_state_dict(torch.load(save_filename))

        candidates_set = generate_candidate_data(data_dir='practice_data',
                                                 cone_file='NGC_2509_cone.csv',
                                                 cluster_file='NGC_2509.tsv',
                                                 candidate_selection_columns=candidate_select_columns,
                                                 probability_threshold=0.9,
                                                 plx_sigma=3.0,
                                                 gmag_max_d=0.2,
                                                 pm_max_d=0.5,
                                                 bp_rp_max_d=0.05)

        evaluate_candidates(candidates_set,
                            model)
