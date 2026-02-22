#!/usr/bin/env bash
# =============================================================
#  experiments/run_all.sh
#  Reproduce all FedCTX-IoT paper experiments.
#  Run from the project root: bash experiments/run_all.sh
# =============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
CONFIG="config/default.yaml"
GPU=0
DATASETS=("ciciot2023" "edge_iiotset" "nbaiot")
ALPHAS=("0.1" "0.3" "0.5" "1.0")
MODELS=("fedavg_cnn" "fedprox_cnn" "fedavg_lstm"
        "fedavg_cnn_transformer" "fedrep_cnn_transformer" "fedctx")

echo "=================================================="
echo "  FedCTX-IoT — Full Experiment Suite"
echo "=================================================="
date

# ── Step 1: Preprocess all datasets ───────────────────────────
echo ""
echo "[ Step 1 ] Preprocessing datasets ..."
for DS in "${DATASETS[@]}"; do
    echo "  → $DS"
    python -m data.preprocessing \
        --dataset   "$DS" \
        --raw_dir   "data/raw/$DS" \
        --out_dir   "data/processed/$DS" \
        --seq_len   32
done

# ── Step 2: Partition (non-IID sweep + IID) ───────────────────
echo ""
echo "[ Step 2 ] Partitioning datasets ..."
for DS in "${DATASETS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        echo "  → $DS | alpha=$ALPHA"
        python -m data.partitioner \
            --processed_dir "data/processed/$DS" \
            --out_dir       "data/partitioned/${DS}/alpha_${ALPHA}" \
            --n_clients     8 \
            --alpha         "$ALPHA"
    done
    # IID partition (alpha=0 triggers IID)
    echo "  → $DS | IID"
    python -m data.partitioner \
        --processed_dir "data/processed/$DS" \
        --out_dir       "data/partitioned/${DS}/iid" \
        --n_clients     8 \
        --alpha         0
done

# ── Step 3: Train all models on all datasets (alpha=0.5) ──────
echo ""
echo "[ Step 3 ] Training all models (alpha=0.5) ..."
for DS in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        OUT="results/${DS}/${MODEL}/alpha_0.5"
        echo "  → $DS | $MODEL"
        python train.py \
            --config        "$CONFIG" \
            --dataset       "$DS" \
            --model         "$MODEL" \
            --partitioned_dir "data/partitioned/${DS}/alpha_0.5" \
            --processed_dir "data/processed/$DS" \
            --out_dir       "$OUT" \
            --gpu           "$GPU"
    done
done

# ── Step 4: Non-IID sensitivity sweep (FedCTX only) ──────────
echo ""
echo "[ Step 4 ] Non-IID sensitivity sweep ..."
for DS in "${DATASETS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUT="results/${DS}/fedctx/alpha_${ALPHA}"
        echo "  → $DS | fedctx | alpha=$ALPHA"
        python train.py \
            --config        "$CONFIG" \
            --dataset       "$DS" \
            --model         fedctx \
            --partitioned_dir "data/partitioned/${DS}/alpha_${ALPHA}" \
            --processed_dir "data/processed/$DS" \
            --out_dir       "$OUT" \
            --gpu           "$GPU"
    done
    # IID
    python train.py \
        --config        "$CONFIG" \
        --dataset       "$DS" \
        --model         fedctx \
        --partitioned_dir "data/partitioned/${DS}/iid" \
        --processed_dir "data/processed/$DS" \
        --out_dir       "results/${DS}/fedctx/iid" \
        --gpu           "$GPU"
done

# ── Step 5: Privacy sweep ──────────────────────────────────────
echo ""
echo "[ Step 5 ] Privacy budget sweep ..."
EPSILONS=("0.1" "0.5" "1.0" "2.0" "3.0" "5.0" "8.0" "10.0")
DS="ciciot2023"
for EPS in "${EPSILONS[@]}"; do
    OUT="results/${DS}/fedctx/privacy_eps_${EPS}"
    echo "  → epsilon=$EPS"
    python train.py \
        --config        "$CONFIG" \
        --dataset       "$DS" \
        --model         fedctx \
        --partitioned_dir "data/partitioned/${DS}/alpha_0.5" \
        --processed_dir "data/processed/$DS" \
        --out_dir       "$OUT" \
        --gpu           "$GPU"
done

# ── Step 6: Evaluate all experiments ─────────────────────────
echo ""
echo "[ Step 6 ] Evaluating all checkpoints ..."
for DS in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        CKPT="results/${DS}/${MODEL}/alpha_0.5/checkpoint_round_0100.pt"
        if [ -f "$CKPT" ]; then
            python evaluate.py \
                --checkpoint      "$CKPT" \
                --dataset         "$DS" \
                --processed_dir   "data/processed/$DS" \
                --partitioned_dir "data/partitioned/${DS}/alpha_0.5" \
                --out_dir         "results/${DS}/${MODEL}/alpha_0.5/eval" \
                --gpu             "$GPU"
        fi
    done
done

# ── Step 7: Generate all paper figures ───────────────────────
echo ""
echo "[ Step 7 ] Generating figures ..."
cd figures
for FIG in fig2_confusion.py fig3_noniid.py fig4_convergence.py \
           fig5_xai.py fig_roc.py fig_tsne.py \
           fig_noniid_dist.py fig_radar.py fig_privacy_tradeoff.py; do
    echo "  → $FIG"
    python "$FIG"
done
cd ..

echo ""
echo "=================================================="
echo "  All experiments complete."
echo "=================================================="
date
