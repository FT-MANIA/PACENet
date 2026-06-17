import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def _safe_div(a, b, eps=1e-12):
    return float(a) / float(b + eps)


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_abnormal_score(y_prob, abnormal_classes=(1, 2)):
    y_prob = _to_numpy(y_prob).astype(float)

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob should be 2D, got shape={y_prob.shape}")

    abnormal_score = y_prob[:, list(abnormal_classes)].sum(axis=1)
    return abnormal_score


def binarize_abnormal_label(y_true, abnormal_classes=(1, 2)):
    y_true = _to_numpy(y_true).astype(int)
    y_bin = np.isin(y_true, list(abnormal_classes)).astype(int)
    return y_bin


def compute_binary_screening_metrics(y_true_bin, abnormal_score, threshold):
    y_true_bin = _to_numpy(y_true_bin).astype(int)
    abnormal_score = _to_numpy(abnormal_score).astype(float)

    y_pred_bin = (abnormal_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1]
    ).ravel()

    sensitivity = _safe_div(tp, tp + fn)
    recall = sensitivity
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        auroc = roc_auc_score(y_true_bin, abnormal_score)
    except Exception:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true_bin, abnormal_score)
    except Exception:
        auprc = np.nan

    return {
        "threshold": float(threshold),

        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),

        "sensitivity": float(sensitivity),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),

        "fpr": float(fpr),
        "fnr": float(fnr),
        "auroc": float(auroc),
        "auprc": float(auprc),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),

        "n": int(len(y_true_bin)),
        "n_abnormal": int(np.sum(y_true_bin == 1)),
        "n_healthy": int(np.sum(y_true_bin == 0)),
        "pred_abnormal_rate": float(np.mean(y_pred_bin == 1)),
    }


def fit_threshold_for_target_sensitivity(
    y_true,
    y_prob,
    target_sensitivity=0.95,
    abnormal_classes=(1, 2)
):
    y_bin = binarize_abnormal_label(y_true, abnormal_classes)
    abnormal_score = get_abnormal_score(y_prob, abnormal_classes)

    if np.sum(y_bin == 1) == 0:
        raise ValueError("No abnormal samples in y_true. Cannot fit sensitivity-constrained threshold.")

    scores = np.asarray(abnormal_score, dtype=float)

    eps = 1e-12
    candidate_thresholds = np.unique(scores)
    candidate_thresholds = np.sort(candidate_thresholds)[::-1]
    candidate_thresholds = np.concatenate([
        candidate_thresholds,
        np.array([scores.min() - eps])
    ])

    best_metrics = None

    for th in candidate_thresholds:
        metrics = compute_binary_screening_metrics(
            y_true_bin=y_bin,
            abnormal_score=abnormal_score,
            threshold=th
        )

        if metrics["sensitivity"] >= target_sensitivity:
            best_metrics = metrics
            break

    if best_metrics is None:
        th = scores.min() - eps
        best_metrics = compute_binary_screening_metrics(
            y_true_bin=y_bin,
            abnormal_score=abnormal_score,
            threshold=th
        )

    best_metrics["target_sensitivity"] = float(target_sensitivity)
    best_metrics["threshold_source"] = "validation"

    return best_metrics

def fit_threshold_for_target_specificity(
    y_true,
    y_prob,
    target_specificity=0.95,
    abnormal_classes=(1, 2)
):
    y_bin = binarize_abnormal_label(y_true, abnormal_classes)
    abnormal_score = get_abnormal_score(y_prob, abnormal_classes)

    if np.sum(y_bin == 0) == 0:
        raise ValueError("No healthy samples in y_true. Cannot fit specificity-constrained threshold.")

    scores = np.asarray(abnormal_score, dtype=float)
    eps = 1e-12

    candidate_thresholds = np.unique(scores)
    candidate_thresholds = np.sort(candidate_thresholds)

    candidate_thresholds = np.concatenate([
        candidate_thresholds,
        np.array([scores.max() + eps])
    ])

    best_metrics = None

    for th in candidate_thresholds:
        metrics = compute_binary_screening_metrics(
            y_true_bin=y_bin,
            abnormal_score=abnormal_score,
            threshold=th
        )

        if metrics["specificity"] >= target_specificity:
            best_metrics = metrics
            break

    if best_metrics is None:
        th = scores.max() + eps
        best_metrics = compute_binary_screening_metrics(
            y_true_bin=y_bin,
            abnormal_score=abnormal_score,
            threshold=th
        )

    best_metrics["target_specificity"] = float(target_specificity)
    best_metrics["threshold_source"] = "validation"

    return best_metrics

def fit_threshold_by_policy(
    y_true,
    y_prob,
    policy="sensitivity",
    target_value=0.95,
    abnormal_classes=(1, 2)
):
    if policy == "sensitivity":
        metrics = fit_threshold_for_target_sensitivity(
            y_true=y_true,
            y_prob=y_prob,
            target_sensitivity=target_value,
            abnormal_classes=abnormal_classes
        )
        metrics["threshold_policy"] = f"sens_ge_{target_value:.2f}"

    elif policy == "specificity":
        metrics = fit_threshold_for_target_specificity(
            y_true=y_true,
            y_prob=y_prob,
            target_specificity=target_value,
            abnormal_classes=abnormal_classes
        )
        metrics["threshold_policy"] = f"spec_ge_{target_value:.2f}"

    else:
        raise ValueError(f"Unsupported threshold policy: {policy}")

    metrics["policy"] = policy
    metrics["target_value"] = float(target_value)

    return metrics

def evaluate_screening_at_threshold(
    y_true,
    y_prob,
    threshold,
    abnormal_classes=(1, 2)
):
    y_bin = binarize_abnormal_label(y_true, abnormal_classes)
    abnormal_score = get_abnormal_score(y_prob, abnormal_classes)

    metrics = compute_binary_screening_metrics(
        y_true_bin=y_bin,
        abnormal_score=abnormal_score,
        threshold=threshold
    )

    return metrics


def thresholded_three_class_prediction(
    y_prob,
    threshold,
    healthy_class=0,
    acld_class=1,
    koa_class=2
):
    y_prob = _to_numpy(y_prob).astype(float)

    abnormal_score = y_prob[:, acld_class] + y_prob[:, koa_class]
    pred = np.full(y_prob.shape[0], healthy_class, dtype=int)

    abnormal_mask = abnormal_score >= threshold
    acld_prob = y_prob[:, acld_class]
    koa_prob = y_prob[:, koa_class]

    pred_abnormal = np.where(acld_prob >= koa_prob, acld_class, koa_class)
    pred[abnormal_mask] = pred_abnormal[abnormal_mask]

    return pred


def evaluate_thresholded_three_class(
    y_true,
    y_prob,
    threshold,
    healthy_class=0,
    acld_class=1,
    koa_class=2
):
    y_true = _to_numpy(y_true).astype(int)

    pred = thresholded_three_class_prediction(
        y_prob,
        threshold,
        healthy_class=healthy_class,
        acld_class=acld_class,
        koa_class=koa_class
    )

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, pred, average="macro", zero_division=0)

    return {
        "thresholded_3class_accuracy": float(acc),
        "thresholded_3class_macro_f1": float(macro_f1),
        "thresholded_3class_macro_recall": float(macro_recall),
        "thresholded_3class_macro_precision": float(macro_precision),
    }


def add_prefix(metrics, prefix):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def mean_std_str(values, digits=4):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return "nan ± nan"

    return f"{values.mean():.{digits}f} ± {values.std(ddof=1 if len(values) > 1 else 0):.{digits}f}"


def summarize_screening_records(records, digits=4):
    df = pd.DataFrame(records)

    summary = {}

    numeric_cols = []
    for c in df.columns:
        if c in ["experiment", "split", "fold"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    for col in numeric_cols:
        summary[f"{col}_str"] = mean_std_str(df[col].values, digits=digits)

    return summary