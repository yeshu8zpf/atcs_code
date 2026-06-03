#!/usr/bin/env bash

set -euo pipefail

echo "[coreset] Installing PyTorch separately"
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[coreset] Installing top-level Python packages"
python -m pip install \
  accelerate==1.0.1 \
  bitsandbytes==0.45.5 \
  datasets==2.19.1 \
  evaluate==0.4.6 \
  llamafactory==0.9.1 \
  matplotlib==3.7.5 \
  nltk==3.8 \
  openai==2.2.0 \
  pandas==2.0.1 \
  peft==0.12.0 \
  pyarrow==17.0.0 \
  sentence-transformers==2.2.2 \
  sentencepiece==0.2.0 \
  spacy==3.7.2 \
  tqdm==4.64.1 \
  transformers==4.46.1 \
  trl==0.9.6 \
  -i https://pypi.org/simple

echo "[coreset] Done."
