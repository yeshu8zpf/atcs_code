#!/bin/bash
set -euo pipefail

bash scripts/run/llama/xsota/fine/prepocess.sh

bash scripts/run/llama/xsota/fine/run1.sh

bash scripts/run/llama/xsota/fine/run2.sh

bash scripts/run/llama/xsota/fine/run3.sh