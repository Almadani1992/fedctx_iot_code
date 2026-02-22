"""
data/partitioner.py
===================
IID and Dirichlet non-IID data partitioning for federated simulation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FederatedPartitioner:
    """
    Partition a dataset across K federated clients using either
    IID uniform sampling or a Dirichlet non-IID distribution.

    Parameters
    ----------
    n_clients : int
        Number of federated clients K.
    alpha : float or None
        Dirichlet concentration parameter. Smaller values produce
        more extreme non-IID distributions. Set to None for IID.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clients: int = 8,
        alpha: Optional[float] = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_clients = n_clients
        self.alpha = alpha
        self.seed  = seed
        self.rng   = np.random.default_rng(seed)

    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split (X, y) into K client datasets.

        Returns
        -------
        clients : list of (X_k, y_k) tuples, length K
        """
        if self.alpha is None:
            return self._iid_partition(X, y)
        return self._dirichlet_partition(X, y)

    def get_distribution_matrix(
        self, y: np.ndarray
    ) -> np.ndarray:
        """
        Return a (K, C) matrix of class proportions per client,
        used for visualisation in fig_noniid_dist.py.
        """
        clients = self.partition(
            np.zeros((len(y), 1)), y
        )
        classes = np.unique(y)
        C = len(classes)
        dist = np.zeros((self.n_clients, C))
        for k, (_, yk) in enumerate(clients):
            for c_idx, c in enumerate(classes):
                dist[k, c_idx] = np.sum(yk == c) / max(len(yk), 1)
        return dist

    # ── Private methods ───────────────────────────────────────────────────

    def _iid_partition(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        idx = self.rng.permutation(len(X))
        splits = np.array_split(idx, self.n_clients)
        clients = [(X[s], y[s]) for s in splits]
        self._log_distribution(clients)
        return clients

    def _dirichlet_partition(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        classes  = np.unique(y)
        n_class  = len(classes)
        client_indices: List[List[int]] = [[] for _ in range(self.n_clients)]

        for c in classes:
            class_idx  = np.where(y == c)[0]
            self.rng.shuffle(class_idx)
            proportions = self.rng.dirichlet(
                np.repeat(self.alpha, self.n_clients)
            )
            proportions = proportions / proportions.sum()
            splits = (proportions * len(class_idx)).astype(int)
            splits[-1] = len(class_idx) - splits[:-1].sum()

            start = 0
            for k, count in enumerate(splits):
                client_indices[k].extend(
                    class_idx[start: start + count].tolist()
                )
                start += count

        clients = []
        for k in range(self.n_clients):
            idx = np.array(client_indices[k], dtype=np.int64)
            self.rng.shuffle(idx)
            clients.append((X[idx], y[idx]))

        self._log_distribution(clients)
        return clients

    def _log_distribution(
        self, clients: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        sizes = [len(c[1]) for c in clients]
        logger.info(
            "Client sizes: min=%d  max=%d  mean=%.0f",
            min(sizes), max(sizes), np.mean(sizes)
        )


def save_partitions(
    clients: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: str | Path,
) -> None:
    """Save each client's data as .npy files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (X_k, y_k) in enumerate(clients):
        client_dir = out_dir / f"client_{k:02d}"
        client_dir.mkdir(exist_ok=True)
        np.save(client_dir / "X.npy", X_k)
        np.save(client_dir / "y.npy", y_k)
    logger.info("Saved %d client partitions to %s", len(clients), out_dir)


def load_partitions(
    partitioned_dir: str | Path,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load all client partitions from disk."""
    partitioned_dir = Path(partitioned_dir)
    client_dirs = sorted(partitioned_dir.glob("client_*"))
    if not client_dirs:
        raise FileNotFoundError(
            f"No client directories found in {partitioned_dir}"
        )
    clients = []
    for d in client_dirs:
        X = np.load(d / "X.npy")
        y = np.load(d / "y.npy")
        clients.append((X, y))
    logger.info("Loaded %d client partitions from %s",
                len(clients), partitioned_dir)
    return clients


# ── CLI entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition a processed dataset for federated training."
    )
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--n_clients",     type=int,   default=8)
    parser.add_argument("--alpha",         type=float, default=0.5,
                        help="Dirichlet alpha. Use 0 for IID.")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    X = np.load(Path(args.processed_dir) / "X_train.npy")
    y = np.load(Path(args.processed_dir) / "y_train.npy")

    alpha = None if args.alpha == 0 else args.alpha
    partitioner = FederatedPartitioner(
        n_clients=args.n_clients, alpha=alpha, seed=args.seed
    )
    clients = partitioner.partition(X, y)
    save_partitions(clients, args.out_dir)


if __name__ == "__main__":
    main()
