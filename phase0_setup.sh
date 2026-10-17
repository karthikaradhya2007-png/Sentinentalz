#!/usr/bin/env bash
# ==============================================================================
# SentinelZ Phase 0 Setup Script (for IBM LinuxONE s390x / JupyterLab)
# ==============================================================================
set -e

echo "=== [1/4] Verifying LinuxONE System Architecture ==="
ARCH=$(uname -m)
echo "System Architecture: $ARCH"
if [ "$ARCH" = "s390x" ]; then
    echo "✅ CONFIRMED: Running natively on IBM LinuxONE / s390x!"
else
    echo "⚠️ NOTE: Not running on s390x (Architecture detected: $ARCH)"
fi

python3 --version
pip3 --version

echo ""
echo "=== [2/4] Setting Up SentinelZ Directory Structure ==="
mkdir -p ~/sentinelz/{data,models,notebooks,api,dashboard,reports,screenshots}
cd ~/sentinelz
echo "✅ Directory structure created at ~/sentinelz:"
ls -la ~/sentinelz

echo ""
echo "=== [3/4] Installing Required Dependencies ==="
pip install --upgrade pip
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn \
            shap fastapi uvicorn streamlit matplotlib seaborn \
            joblib requests

echo ""
echo "=== [4/4] Verifying Python Environment ==="
python3 -c "
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb
import fastapi
import streamlit
print('✅ Python environment verification SUCCESS!')
print('  Pandas:', pd.__version__)
print('  NumPy:', np.__version__)
print('  Scikit-Learn:', sklearn.__version__)
print('  LightGBM:', lgb.__version__)
print('  FastAPI:', fastapi.__version__)
"

echo ""
echo "🎉 Phase 0 Environment Setup Complete! Ready for Phase 1."
