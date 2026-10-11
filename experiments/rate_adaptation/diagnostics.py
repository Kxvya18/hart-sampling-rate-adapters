"""Training-only balancing and explicit class/subject evaluation diagnostics."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def training_order(indices, labels, rng, sampling):
    if sampling == "uniform":
        return rng.permutation(indices)
    _, inverse, counts = np.unique(labels[indices], return_inverse=True, return_counts=True)
    # Square-root balancing limits duplication of extremely rare training classes.
    weights = 1 / np.sqrt(counts[inverse])
    return rng.choice(indices, size=len(indices), replace=True, p=weights / weights.sum())


def class_diagnostics(y, probabilities, classes):
    predicted = classes[probabilities.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=classes, zero_division=0)
    return [{"class": int(c), "support": int(support[i]),
             "predicted_count": int(np.sum(predicted == c)),
             "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i])} for i, c in enumerate(classes)]
