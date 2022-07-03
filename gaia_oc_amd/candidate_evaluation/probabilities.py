import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.candidate_evaluation.sampling import mean_and_cov_matrix, sample_sources
from gaia_oc_amd.data_preparation.datasets import deep_sets_eval_dataset
from gaia_oc_amd.data_preparation.utils import normalize
from gaia_oc_amd.data_preparation.features import add_features


def predict_member(model, data, batch_size=1024):
    n_candidates = data.size(0)
    n_batches = n_candidates // batch_size
    predictions = []
    start = 0

    for _ in range(n_batches):
        predictions.append(softmax(model(data[start: start + batch_size]), 1).detach().numpy()[:, 1] > 0.5)
        start += batch_size
    predictions.append(softmax(model(data[start:]), 1).detach().numpy()[:, 1] > 0.5)

    return np.concatenate(predictions)


def calculate_probabilities(eval_sources, model, sources, features, n_samples, size_support_set=5, seed=42):
    sample_predictions = np.zeros((n_samples, len(eval_sources)))

    means, cov_matrices = mean_and_cov_matrix(eval_sources)
    all_data = sources.all_sources[features.train_features]
    members = normalize(sources.members.hp()[features.train_features], all_data=all_data).to_numpy()

    model.eval()
    np.random.seed(seed)
    for sample_idx in tqdm(range(n_samples), total=n_samples, desc='Evaluating candidate samples'):
        sample = sample_sources(eval_sources, means, cov_matrices)
        add_features([sample], features.train_feature_functions, features.train_feature_labels)
        sample = normalize(sample[features.train_features], all_data=all_data).to_numpy()
        sample_dataset = deep_sets_eval_dataset(sample, members, size_support_set)
        sample_predictions[sample_idx] = predict_member(model, sample_dataset)

    probabilities = np.mean(sample_predictions, axis=0)

    return probabilities
