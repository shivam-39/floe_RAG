#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

python src/benchmark.py \
    --dataset-name scifact \
    --dataset-url https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
    --dataset-dir benchmark_data/scifact \
    --index-dir benchmark_index/scifact \
    --results-path benchmark_results/scifact.json \
    --split train \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --embedding-device cpu \
    --chunk-size 400 \
    --chunk-overlap 80 \
    --k-values 1,5,10
