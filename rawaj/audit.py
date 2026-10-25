"""Data audit for Rawaj Pay transfer extracts."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.metrics import roc_auc_score


def _missing_mask(series: pd.Series) -> pd.Series:
    """Treat NaN, None, and empty or whitespace-only strings as missing."""
    blank_strings = series.map(
        lambda value: isinstance(value, str) and value.strip() == ""
    )
    return series.isna() | blank_strings


def _is_constant_including_missing(
    series: pd.Series,
    missing: pd.Series,
) -> bool:
    """Check whether there is at most one distinct value, counting missing."""
    distinct_non_missing = int(series.loc[~missing].nunique(dropna=True))
    has_missing = bool(missing.any())
    return distinct_non_missing + int(has_missing) <= 1


def audit(
    df: pd.DataFrame,
    *,
    target: str,
    id_col: str,
    time_col: str,
) -> dict:
    """Audit a transfer dataset before model training."""

    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)

    n_rows = int(len(df))

    missing_masks = {
        column: _missing_mask(df[column])
        for column in df.columns
    }

    missing = {
        column: float(mask.sum() / n_rows) if n_rows else 0.0
        for column, mask in missing_masks.items()
    }

    # Repeated non-missing IDs.
    id_values = df.loc[~missing_masks[id_col], id_col]
    repeated_ids = (
        id_values[id_values.duplicated(keep=False)]
        .drop_duplicates()
        .tolist()
    )

    try:
        duplicate_ids = sorted(repeated_ids)
    except TypeError:
        duplicate_ids = sorted(repeated_ids, key=lambda value: str(value))

    # Rows that exactly repeat an earlier row.
    duplicate_rows = int(df.duplicated(keep="first").sum())

    # Constant columns exclude the target; missing values are ignored here.
    constant_columns = []

    for column in df.columns:
        if column == target:
            continue

        observed = df.loc[~missing_masks[column], column]

        if observed.nunique(dropna=True) <= 1:
            constant_columns.append(column)

    constant_columns.sort()

    # Label coverage and positive rate.
    target_missing = missing_masks[target]
    labelled = df.loc[~target_missing, target]
    unlabelled_rows = int(target_missing.sum())

    positive_rate = (
        float((labelled == 1).sum() / len(labelled))
        if len(labelled)
        else 0.0
    )

    # Parse timestamps; blank timestamps are missing, not malformed.
    time_missing = missing_masks[time_col]

    parsed_times = pd.to_datetime(
        df[time_col].where(~time_missing),
        errors="coerce",
    )

    bad_timestamps = int(
        ((~time_missing) & parsed_times.isna()).sum()
    )

    valid_times = parsed_times.dropna()

    if len(valid_times):
        time_range = [
            valid_times.min().isoformat(),
            valid_times.max().isoformat(),
        ]
    else:
        time_range = [None, None]

    # Leakage checks only use labelled rows with binary labels.
    binary_labelled_mask = (
        (~target_missing)
        & df[target].isin([0, 1])
    )
    y_all = df.loc[binary_labelled_mask, target].astype(int)

    leakage_suspects = []

    if len(y_all) >= 20 and y_all.nunique() == 2:
        excluded = {target, id_col, time_col}

        for column in df.columns:
            if column in excluded:
                continue

            column_missing = missing_masks[column]

            # Exclude columns constant across all rows when missing counts
            # as a possible distinct value.
            if _is_constant_including_missing(
                df[column],
                column_missing,
            ):
                continue

            values = df.loc[binary_labelled_mask, column]
            missing_on_labelled = column_missing.loc[
                binary_labelled_mask
            ]

            # (a) Value-based leakage for numeric/boolean columns.
            if (
                is_numeric_dtype(df[column].dtype)
                or is_bool_dtype(df[column].dtype)
            ):
                observed_mask = ~missing_on_labelled

                if int(observed_mask.sum()) >= 20:
                    y_observed = y_all.loc[observed_mask]
                    x_observed = values.loc[observed_mask]

                    if y_observed.nunique() == 2:
                        try:
                            auc = float(
                                roc_auc_score(y_observed, x_observed)
                            )

                            if auc >= 0.98 or auc <= 0.02:
                                leakage_suspects.append(column)
                                continue
                        except (ValueError, TypeError):
                            pass

            # (b) Missingness-based leakage for any column type.
            missing_indicator = missing_on_labelled.astype(int)

            if missing_indicator.nunique() == 2:
                try:
                    missing_auc = float(
                        roc_auc_score(y_all, missing_indicator)
                    )

                    if missing_auc >= 0.98 or missing_auc <= 0.02:
                        leakage_suspects.append(column)
                except (ValueError, TypeError):
                    pass

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": sorted(set(constant_columns)),
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(set(leakage_suspects)),
    }