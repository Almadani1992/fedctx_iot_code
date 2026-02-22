"""
evaluate.py
===========
Load a trained checkpoint and run full evaluation:
  - Classification metrics (all datasets)
  - Per-class breakdown
  - Membership inference attack
  - XAI fidelity (attention vs SHAP Pearson r)
  - Generate all paper figures

Usage
-----
python evaluate.py \
    --checkpoint results/ciciot2023/checkpoint_round_0100.pt \
    --dataset ciciot2023 \
    --processed_dir data/processed/ciciot2023 \
    --partitioned_dir data/partitioned/ciciot2023 \
    --out_dir results/ciciot2023/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from data.preprocessing import IoTPreprocessor
from models.fedctx_model import build_model_from_config
from evaluation.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    MembershipInferenceAttack,
    xai_fidelity,
)
from xai.attention_xai import AttentionAttributor
from xai.shap_xai import FederatedSHAP

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--config",         default="config/default.yaml")
    parser.add_argument("--dataset",        required=True)
    parser.add_argument("--processed_dir",  required=True)
    parser.add_argument("--partitioned_dir",required=True)
    parser.add_argument("--out_dir",        default="eval_results")
    parser.add_argument("--gpu",            type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["name"] = args.dataset

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ── Load test data ────────────────────────────────────────────────────
    processed = Path(args.processed_dir)
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")
    in_features = X_test.shape[-1]

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model_from_config(cfg, in_features).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint loaded from round %d", ckpt.get("round", -1))

    # ── Classification metrics ────────────────────────────────────────────
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)

    metrics = compute_metrics(y_test, preds, probs)
    logger.info("Test metrics: %s", metrics)

    preprocessor = IoTPreprocessor.load(args.dataset, processed)
    class_names  = list(preprocessor.label_encoder.classes_)
    per_class    = compute_per_class_metrics(y_test, preds, class_names)

    # ── XAI: Pathway 1 ───────────────────────────────────────────────────
    attributor = AttentionAttributor(model)
    sample     = X_t[:256]
    attn_imp   = attributor.compute(sample)
    logger.info("Attention importance computed. Top feature idx: %d",
                attn_imp.argmax())

    # ── XAI: Pathway 2 ───────────────────────────────────────────────────
    background   = X_test[:100]
    shap_module  = FederatedSHAP(model, background, device, add_dp_noise=True)
    shap_vals    = shap_module.explain(X_test[0])
    fidelity     = xai_fidelity(attn_imp, shap_vals)
    logger.info("XAI fidelity — Pearson r=%.4f  p=%.4e",
                fidelity["pearson_r"], fidelity["p_value"])

    # ── Membership inference ──────────────────────────────────────────────
    X_train = np.load(Path(args.partitioned_dir) / "client_00" / "X.npy")
    y_train = np.load(Path(args.partitioned_dir) / "client_00" / "y.npy")

    def model_fn(X: np.ndarray) -> np.ndarray:
        xt = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.softmax(model(xt), dim=-1).cpu().numpy()

    mi_attack = MembershipInferenceAttack(n_shadow_models=4)
    mi_result = mi_attack.evaluate(
        model_fn,
        X_train[:1000], y_train[:1000],
        X_test[:1000],  y_test[:1000],
    )
    logger.info("MI attack: success=%.4f  advantage=%.4f",
                mi_result["success_rate"], mi_result["advantage"])

    # ── Save all results ──────────────────────────────────────────────────
    results = {
        "metrics":     metrics,
        "per_class":   per_class,
        "xai_fidelity": fidelity,
        "mi_attack":   mi_result,
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(out_dir / "attn_importance.npy", attn_imp)
    np.save(out_dir / "shap_values.npy",     shap_vals)
    np.save(out_dir / "y_pred.npy",          preds)
    np.save(out_dir / "y_prob.npy",          probs)

    logger.info("Evaluation complete. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
