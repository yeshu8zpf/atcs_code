#!/usr/bin/env bash

set -euo pipefail

echo "[opencompass] Installing PyTorch separately"
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[opencompass] Installing top-level Python packages"
python -m pip install \
  opencompass==0.5.1 \
  vllm==0.5.3.post1 \
  transformers==4.46.1 \
  datasets==4.8.5 \
  mmengine-lite==0.10.7 \
  nltk==3.8 \
  evaluate==0.4.6 \
  sacrebleu==2.5.1 \
  rouge-score==0.1.2 \
  -i https://pypi.org/simple

echo "[opencompass] Downloading NLTK resources commonly needed by evaluation tasks"
python -m nltk.downloader \
  punkt \
  punkt_tab \
  stopwords \
  wordnet \
  omw-1.4 \
  averaged_perceptron_tagger \
  averaged_perceptron_tagger_eng \
  brown

echo "[opencompass] Done."
