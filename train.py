"""
train.py
=========
Main entry point for FedCTX-IoT federated training.

Usage
-----
python train.py --config config/default.yaml \
                --dataset ciciot2023 \
                --partitioned_dir data/partitioned/ciciot2023 \
                --out_dir results/ciciot2023
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data.partitioner import load_partitions
from models.fedctx_model import build_model_from_config
from federated.client import FederatedClient
from federated.server import FederatedServer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FedCTX-IoT federated learning framework."
    )
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--dataset", required=True,
                        choices=["ciciot2023", "edge_iiotset", "nbaiot"])
    parser.add_argument("--partitioned_dir", required=True,
                        help="Directory with client partitions.")
    parser.add_argument("--processed_dir", required=True,
                        help="Directory with processed test set.")
    parser.add_argument("--out_dir", default="results",
                        help="Output directory for checkpoints and logs.")
    parser.add_argument("--model",
                        default="fedctx",
                        choices=["fedavg_cnn", "fedprox_cnn", "fedavg_lstm",
                                 "fedavg_cnn_transformer",
                                 "fedrep_cnn_transformer", "fedctx"],
                        help="Model / FL algorithm to train.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (-1 for CPU).")
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir = Path(args.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["name"] = args.dataset

    set_seed(cfg["dataset"]["random_seed"])

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────
    clients_data = load_partitions(args.partitioned_dir)
    n_clients    = len(clients_data)
    cfg["federated"]["n_clients"] = n_clients

    X_test = np.load(Path(args.processed_dir) / "X_test.npy")
    y_test = np.load(Path(args.processed_dir) / "y_test.npy")

    in_features = clients_data[0][0].shape[-1]
    logger.info(
        "Loaded %d clients | in_features=%d | test_size=%d",
        n_clients, in_features, len(X_test)
    )

    # ── Model ─────────────────────────────────────────────────────────────
    global_model = build_model_from_config(cfg, in_features)
    param_info   = global_model.count_parameters()
    logger.info(
        "Model params: total=%d | backbone=%d (%.1f%% transmitted)",
        param_info["total_params"],
        param_info["backbone_params"],
        param_info["transmitted_pct"],
    )

    # ── Clients ───────────────────────────────────────────────────────────
    import copy
    clients = [
        FederatedClient(
            client_id = k,
            model     = copy.deepcopy(global_model),
            X         = clients_data[k][0],
            y         = clients_data[k][1],
            cfg       = cfg,
            device    = device,
        )
        for k in range(n_clients)
    ]

    # ── Server ────────────────────────────────────────────────────────────
    server = FederatedServer(
        global_model = global_model,
        clients      = clients,
        X_test       = X_test,
        y_test       = y_test,
        cfg          = cfg,
        out_dir      = args.out_dir,
        device       = device,
    )
    server.run()


if __name__ == "__main__":
    main()
