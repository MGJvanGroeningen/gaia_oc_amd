import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.candidate_evaluation.sampling import candidate_distribution, sample_candidates
from gaia_oc_amd.data_preparation.datasets import candidate_dataset


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


def candidate_probabilities(model, sources, features, n_samples, size_support_set=5, seed=42):
    size_support_set = min(len(sources.members.hp()), size_support_set)
    predictions = np.zeros((n_samples, len(sources.candidates)))

    model.eval()
    means, cov_matrices = candidate_distribution(sources.candidates)
    np.random.seed(seed)
    for sample_idx in tqdm(range(n_samples), total=n_samples, desc='Evaluating candidate samples'):
        candidate_sample = sample_candidates(sources.candidates, means, cov_matrices)
        candidates_dataset = candidate_dataset(candidate_sample, sources, features, size_support_set)
        predictions[sample_idx] = predict_member(model, candidates_dataset)

    probabilities = np.mean(predictions, axis=0)

    return probabilities
