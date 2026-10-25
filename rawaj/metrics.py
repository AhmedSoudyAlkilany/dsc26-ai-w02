"""Decision metrics for a mule-risk score."""

from __future__ import annotations

import math
# pyrefly: ignore[missing-import]
import numpy as np


def _validated_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return one-dimensional NumPy arrays."""

    try:
        y = np.asarray(y_true)
        s = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y_true and scores must contain valid numeric values"
        ) from exc

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")

    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")

    if len(y) == 0:
        raise ValueError("y_true and scores must not be empty")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("every label must be 0 or 1")

    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")

    return y.astype(int, copy=False), s


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return TP, FP, TN, and FN; scores >= threshold are flagged."""

    y, s = _validated_arrays(y_true, scores)

    flagged = s >= threshold
    positive = y == 1
    negative = ~positive

    return {
        "tp": int(np.count_nonzero(flagged & positive)),
        "fp": int(np.count_nonzero(flagged & negative)),
        "tn": int(np.count_nonzero((~flagged) & negative)),
        "fn": int(np.count_nonzero((~flagged) & positive)),
    }


def precision_recall_at(
    y_true,
    scores,
    threshold: float,
) -> tuple[float, float]:
    """Return (precision, recall), using zero when a denominator is zero."""

    counts = confusion_at(y_true, scores, threshold)

    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]

    precision = (
        counts["tp"] / predicted_positive
        if predicted_positive
        else 0.0
    )

    recall = (
        counts["tp"] / actual_positive
        if actual_positive
        else 0.0
    )

    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision among the k highest scores; ties preserve input order."""

    y, s = _validated_arrays(y_true, scores)

    if k <= 0:
        raise ValueError("k must be greater than zero")

    k_eff = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")

    return float(np.mean(y[order[:k_eff]] == 1))


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Calculate FP * cost_fp + FN * cost_fn."""

    counts = confusion_at(y_true, scores, threshold)

    return float(
        counts["fp"] * cost_fp
        + counts["fn"] * cost_fn
    )


def _threshold_statistics(y: np.ndarray, s: np.ndarray):
    """Yield (threshold, precision, cost) for each distinct score.

    Scores are sorted once. Equal scores are processed as one group because
    all rows with score >= threshold must be flagged together.
    """

    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    sorted_y = y[order]

    total_positive = int(np.count_nonzero(y == 1))
    total_negative = len(y) - total_positive

    tp = 0
    fp = 0
    i = 0

    while i < len(y):
        threshold = float(sorted_scores[i])
        j = i

        # Include every row tied at this score.
        while j < len(y) and sorted_scores[j] == sorted_scores[i]:
            if sorted_y[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        predicted_positive = tp + fp
        precision = tp / predicted_positive if predicted_positive else 0.0

        yield threshold, float(precision), tp, fp

        i = j


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Find the minimum-cost threshold; ties choose the highest threshold."""

    y, s = _validated_arrays(y_true, scores)

    total_positive = int(np.count_nonzero(y == 1))
    total_negative = len(y) - total_positive

    # Flag nothing: every positive is a false negative.
    best_threshold_value = math.inf
    best_cost = total_positive * cost_fn

    for threshold, _, tp, fp in _threshold_statistics(y, s):
        fn = total_positive - tp
        cost = fp * cost_fp + fn * cost_fn

        # Descending thresholds and strict improvement preserve the
        # highest threshold when costs tie.
        if cost < best_cost:
            best_cost = cost
            best_threshold_value = threshold

    return float(best_threshold_value)


def threshold_for_precision(
    y_true,
    scores,
    min_precision: float,
) -> float | None:
    """Return the lowest distinct score meeting min_precision, else None."""

    y, s = _validated_arrays(y_true, scores)

    result = None

    # Thresholds are yielded in descending order.
    # Keep updating the result so the final one is the lowest
    # threshold that meets the required precision.
    for threshold, precision, _, _ in _threshold_statistics(y, s):
        if precision >= min_precision:
            result = threshold

    return float(result) if result is not None else None