"""Development-only confidence baselines; this module performs no certification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .io import sha256_file, write_json
from .metrics import confidence_scores, correctness, rank_order, retrieval_metrics, selective_metrics


def fit_temperature(scores, keys, truths) -> dict:
    """Fit multiclass NLL on present fit_selector molecules only."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = []
    for row, truth in zip(keys, truths, strict=True):
        match = np.flatnonzero(np.asarray(row).astype(str) == str(truth))
        if len(match) != 1:
            raise ValueError("Temperature fitting requires exactly one true candidate per query")
        targets.append(int(match[0]))
    targets = np.asarray(targets)
    def nll(log_t):
        logits = scores / np.exp(log_t)
        return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(logits)), targets]))
    result = minimize_scalar(nll, bounds=(-np.log(256), np.log(256)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fit did not converge")
    return {"temperature": float(np.exp(result.x)), "nll_before": nll(0), "nll_after": float(result.fun),
            "fit_role": "fit_selector", "fit_scenario": "present", "n_molecules": len(targets),
            "certified": False}


def select_development_threshold(present_score, absent_score, present_correct, config) -> dict:
    """Choose from a development grid; no confidence guarantee is implied."""
    ps, aus = np.asarray(present_score), np.asarray(absent_score)
    correct = np.asarray(present_correct, dtype=bool)
    points = int(config.get("grid_points", 21))
    grid = np.unique(np.quantile(np.concatenate([ps, aus]), np.linspace(0, 1, max(2, points - 1))))
    candidates = []
    for threshold in grid:
        accepted = ps >= threshold
        n = int(accepted.sum())
        risk = float((~correct[accepted]).mean()) if n else None
        far = float((aus >= threshold).mean())
        if n and risk <= config.get("target_present_risk", .1) and far <= config.get("target_absent_far", .05):
            candidates.append((n, float(threshold), risk, far))
    if not candidates:
        return {"threshold": None, "reject_all": True, "fit_present_acceptance_rate": 0.0,
                "fit_present_risk": None, "fit_unknown_far": 0.0, "certified": False,
                "reason": "No nonempty development acceptance set satisfies empirical targets"}
    n, threshold, risk, far = max(candidates, key=lambda c: (c[0], c[1]))
    return {"threshold": threshold, "reject_all": False, "fit_present_acceptance_rate": n / len(ps),
            "fit_present_risk": risk, "fit_unknown_far": far, "certified": False}


def run_selection_smoke(npz_path, score_path, output_dir, config) -> dict:
    if config.get("formal_calibration", False):
        raise ValueError("Smoke entrypoint cannot perform formal calibration")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(npz_path, allow_pickle=False) as data, np.load(score_path, allow_pickle=False) as saved:
        current_hash = sha256_file(npz_path)
        bound_hash = np.asarray(saved["input_npz_sha256"])
        if bound_hash.size != 1 or str(bound_hash.reshape(-1)[0]) != current_hash:
            raise ValueError("Scores are not bound to the current spectrum/candidate NPZ")
        for prefix, expected in [("fit", "fit_selector"), ("eval", "dev_ranker")]:
            role = np.asarray(data[f"{prefix}_role"])
            if role.size != 1 or str(role.reshape(-1)[0]) != expected:
                raise ValueError("Sealed or unexpected data role in smoke selector")
            if not np.array_equal(saved[f"{prefix}_query_ids"], data[f"{prefix}_query_ids"]):
                raise ValueError("Score rows and query identities are misaligned")
        report = {"scope": "100-query engineering smoke; not a trained-model benchmark or certification",
                  "formal_calibration_accessed": False, "test_accessed": False,
                  "npz_sha256": current_hash, "scores_sha256": sha256_file(score_path),
                  "config": config, "models": {}}
        csv_path = output_dir / "selection_per_query.csv"
        with csv_path.open("w", newline="") as handle:
            fields = ["model", "selector", "role", "scenario", "query_id", "truth_key", "top1_key", "correct", "selection_score", "accepted"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model in ("mlp", "nn"):
                fk, ft = data["fit_keys_present"], data["fit_truth_keys"]
                fitp, fita = saved[f"{model}_fit_present"], saved[f"{model}_fit_absent"]
                temp = fit_temperature(fitp, fk, ft)
                fit_features = {"present": confidence_scores(fitp), "absent": confidence_scores(fita)}
                for scenario, scores in [("present", fitp), ("absent", fita)]:
                    fit_features[scenario]["temperature_probability"] = confidence_scores(scores, temp["temperature"])["max_probability"]
                fit_correct = correctness(fitp, fk, ft, data["fit_query_ids"])
                model_report = {"temperature": temp, "retrieval_present": {}, "selectors": {}}
                eval_features = {}
                for scenario in ("present", "absent"):
                    scores = saved[f"{model}_eval_{scenario}"]
                    eval_features[scenario] = confidence_scores(scores)
                    eval_features[scenario]["temperature_probability"] = confidence_scores(scores, temp["temperature"])["max_probability"]
                for prefix in ("fit", "eval"):
                    model_report["retrieval_present"][prefix] = retrieval_metrics(
                        saved[f"{model}_{prefix}_present"], data[f"{prefix}_keys_present"],
                        data[f"{prefix}_truth_keys"], data[f"{prefix}_query_ids"])
                for method in fit_features["present"]:
                    rule = select_development_threshold(fit_features["present"][method], fit_features["absent"][method], fit_correct, config)
                    model_report["selectors"][method] = {"rule": rule}
                    for prefix, features, role in [("fit", fit_features, "fit_selector"), ("eval", eval_features, "dev_ranker")]:
                        vectors = []
                        for scenario in ("present", "absent"):
                            scores = saved[f"{model}_{prefix}_{scenario}"]
                            keys = data[f"{prefix}_keys_{scenario}"]
                            truths, ids = data[f"{prefix}_truth_keys"], data[f"{prefix}_query_ids"]
                            correct = correctness(scores, keys, truths, ids)
                            absent = np.full(len(ids), scenario == "absent")
                            values = features[scenario][method]
                            accepted = np.zeros(len(ids), dtype=bool) if rule["reject_all"] else values >= rule["threshold"]
                            vectors.append((correct, accepted, absent))
                            for row_index, qid in enumerate(ids):
                                top = rank_order(scores[row_index], keys[row_index], str(qid))[0]
                                writer.writerow(dict(zip(fields, [model, method, role, scenario, str(qid), str(truths[row_index]),
                                    str(keys[row_index, top]), bool(correct[row_index]), float(values[row_index]), bool(accepted[row_index])], strict=True)))
                        # The paired 50/50 concatenation is descriptive, not 2n independent molecules.
                        model_report["selectors"][method][prefix] = selective_metrics(*[
                            np.concatenate([v[j] for v in vectors]) for j in range(3)])
                report["models"][model] = model_report
        report["elapsed_seconds"] = time.monotonic() - started
        report["per_query_sha256"] = sha256_file(csv_path)
        write_json(output_dir / "selection_results.json", report)
    return report
