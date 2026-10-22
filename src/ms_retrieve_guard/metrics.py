"""Identity-based metrics; deterministic ties and explicit undefined denominators."""

from __future__ import annotations

import hashlib
import numpy as np
from scipy.special import softmax
from scipy.stats import beta


def rank_order(scores, candidate_keys, query_id: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    keys = np.asarray(candidate_keys).astype(str)
    if scores.ndim != 1 or keys.shape != scores.shape or scores.size == 0:
        raise ValueError("Expected a nonempty aligned score/key vector")
    if not np.isfinite(scores).all() or len(set(keys)) != len(keys):
        raise ValueError("Nonfinite scores or duplicate candidate identities")
    # Never perturb scores. Hashing is independent of candidate position and truth.
    return np.asarray(sorted(range(len(scores)), key=lambda j: (
        -scores[j], hashlib.sha256(f"tie-v1|{query_id}|{keys[j]}".encode()).digest())))


def correctness(scores, candidate_keys, truth_keys, query_ids):
    values = []
    for row, keys, truth, query in zip(scores, candidate_keys, truth_keys, query_ids, strict=True):
        order = rank_order(row, keys, str(query))
        values.append(str(keys[order[0]]) == str(truth))
    return np.asarray(values, dtype=bool)


def retrieval_metrics(scores, candidate_keys, truth_keys, query_ids) -> dict:
    ranks = []
    for row, keys, truth, query in zip(scores, candidate_keys, truth_keys, query_ids, strict=True):
        matches = np.flatnonzero(np.asarray(keys).astype(str) == str(truth))
        if len(matches) != 1:
            raise ValueError("Closed-set retrieval metrics require exactly one true identity")
        order = rank_order(row, keys, str(query))
        ranks.append(int(np.flatnonzero(order == matches[0])[0]) + 1)
    if not ranks:
        return {"n": 0, "recall_at_1": None, "recall_at_5": None, "recall_at_20": None, "mrr": None}
    ranks = np.asarray(ranks)
    return {"n": len(ranks), **{f"recall_at_{k}": float(np.mean(ranks <= k)) for k in (1, 5, 20)},
            "mrr": float(np.mean(1.0 / ranks))}


def selective_metrics(correct, accepted, absent) -> dict:
    correct, accepted, absent = [np.asarray(a, dtype=bool) for a in (correct, accepted, absent)]
    if correct.ndim != 1 or correct.shape != accepted.shape or correct.shape != absent.shape:
        raise ValueError("Metric arrays must be aligned 1D vectors")
    if np.any(correct & absent):
        raise ValueError("Absent queries cannot have a correct in-pool prediction")
    present = ~absent
    def ratio(a, b):
        return float(a / b) if b else None
    n, na, npres, nabs = len(correct), int(accepted.sum()), int(present.sum()), int(absent.sum())
    nap = int((accepted & present).sum())
    ep = int((accepted & present & ~correct).sum())
    ea = int((accepted & absent).sum())
    errors = int((accepted & ~correct).sum())
    return {"n_queries": n, "n_present": npres, "n_absent": nabs, "n_accepted": na,
            "n_present_accepted": nap, "n_present_accepted_errors": ep, "n_absent_accepted": ea,
            "acceptance_rate": ratio(na, n), "accepted_error_rate": ratio(errors, na),
            "present_acceptance_rate": ratio(nap, npres), "present_accepted_error_rate": ratio(ep, nap),
            "unknown_far": ratio(ea, nabs)}


def confidence_scores(scores, temperature: float = 1.0) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Finite scores with at least two candidates are required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    ordered = np.sort(scores, axis=1)
    prob = softmax(scores / temperature, axis=1)
    entropy = -(prob * np.log(np.maximum(prob, np.finfo(float).tiny))).sum(1) / np.log(scores.shape[1])
    return {"max_score": ordered[:, -1], "margin": ordered[:, -1] - ordered[:, -2],
            "max_probability": prob.max(1), "negative_entropy": -entropy}


def binomial_upper(errors: int, n: int, delta: float) -> float | None:
    """One-sided Clopper-Pearson upper bound for a FIXED rule and valid sampling model."""
    if not isinstance(n, (int, np.integer)) or not isinstance(errors, (int, np.integer)):
        raise ValueError("Counts must be integers")
    if n < 0 or errors < 0 or errors > n or not 0 < delta < 1:
        raise ValueError("Invalid counts or delta")
    if n == 0:
        return None
    return 1.0 if errors == n else float(beta.ppf(1 - delta, errors + 1, n - errors))
