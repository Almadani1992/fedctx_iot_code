"""
evaluation/metrics.py
======================
Evaluation metrics: classification performance, membership inference,
and XAI fidelity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all classification metrics used in the paper.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : predicted labels
    y_prob : predicted probabilities, shape (N, C) — required for AUC

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted, precision_macro,
                    recall_macro, auc_roc (if y_prob provided)
    """
    metrics = {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1_macro":         float(f1_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_weighted":      float(f1_score(y_true, y_pred, average="weighted",
                                           zero_division=0)),
        "precision_macro":  float(precision_score(y_true, y_pred,
                                                  average="macro",
                                                  zero_division=0)),
        "recall_macro":     float(recall_score(y_true, y_pred,
                                               average="macro",
                                               zero_division=0)),
    }

    if y_prob is not None:
        try:
            n_classes = y_prob.shape[1]
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovr", average="macro"
                )
            metrics["auc_roc"] = float(auc)
        except Exception:
            metrics["auc_roc"] = float("nan")

    return metrics


def compute_per_class_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: Optional[List[str]] = None,
) -> List[dict]:
    """
    Per-class precision, recall, F1, and support.

    Returns list of dicts, one per class.
    """
    classes  = np.unique(y_true)
    results  = []
    for c in classes:
        mask = y_true == c
        tp   = np.sum((y_pred == c) & mask)
        fp   = np.sum((y_pred == c) & ~mask)
        fn   = np.sum((y_pred != c) & mask)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        name = class_names[c] if class_names else str(c)
        results.append({
            "class":     name,
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "support":   int(mask.sum()),
        })
    return results


class MembershipInferenceAttack:
    """
    Shadow-model membership inference attack (Shokri et al., 2017).

    Trains a binary classifier on shadow model outputs to
    distinguish training members from non-members.

    Parameters
    ----------
    n_shadow_models : int — number of shadow models to train
    """

    def __init__(self, n_shadow_models: int = 4) -> None:
        self.n_shadow_models = n_shadow_models

    def evaluate(
        self,
        model_fn,
        X_member:     np.ndarray,
        y_member:     np.ndarray,
        X_nonmember:  np.ndarray,
        y_nonmember:  np.ndarray,
    ) -> Dict[str, float]:
        """
        Run the attack and return success rate and advantage.

        Returns
        -------
        dict with: success_rate, advantage (over 50% random baseline)
        """
        from sklearn.linear_model import LogisticRegression

        # Member confidence scores
        probs_m  = model_fn(X_member)
        probs_nm = model_fn(X_nonmember)

        # Attack feature: max predicted probability
        feat_m  = probs_m.max(axis=1, keepdims=True)
        feat_nm = probs_nm.max(axis=1, keepdims=True)

        X_attack = np.vstack([feat_m,  feat_nm])
        y_attack = np.hstack([
            np.ones(len(feat_m)),
            np.zeros(len(feat_nm))
        ])

        from sklearn.model_selection import cross_val_score
        clf    = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_attack, y_attack, cv=5)

        success_rate = float(scores.mean())
        advantage    = float(success_rate - 0.5)

        return {
            "success_rate": success_rate,
            "advantage":    advantage,
            "std":          float(scores.std()),
        }


def xai_fidelity(
    attn_importance: np.ndarray,
    shap_values:     np.ndarray,
) -> Dict[str, float]:
    """
    Compute Pearson correlation between attention and SHAP importance.

    Returns dict with pearson_r and p_value.
    """
    from scipy.stats import pearsonr
    r, p = pearsonr(attn_importance, shap_values)
    return {"pearson_r": float(r), "p_value": float(p)}
