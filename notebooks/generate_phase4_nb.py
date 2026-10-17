import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🛡️ SentinelZ: Phase 4 — High-Concurrency Streaming Simulation\n",
    "**IBM Z Datathon 2026 | Theme 1: Real-Time AI for Critical Decisions**  \n",
    "*Platform:* **IBM LinuxONE Community Cloud (s390x)**  \n",
    "*Service:* **SentinelZ In-Flight Fraud Prevention API (Port 8000)**\n",
    "\n",
    "---\n",
    "\n",
    "### Objectives for Phase 4:\n",
    "1. **Stream Live Transactions:** Emulate real-time payment gateway traffic (ISO 20022 / PaySim).\n",
    "2. **Real-Time API Interception:** Stream transactions to `http://127.0.0.1:8000/score`.\n",
    "3. **SLA Verification:** Prove sub-millisecond execution (`< 5ms` banking SLA).\n",
    "4. **Analyze Latency Profile:** Calculate P50, P90, P99 tail latency and throughput (TPS).\n",
    "5. **Generate Visual Proof:** Export `reports/streaming_latency_benchmark.png` for submission."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.0: Health Check of SentinelZ Real-Time API\n",
    "import requests, time, os, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "API_URL = \"http://127.0.0.1:8000\"\n",
    "try:\n",
    "    resp = requests.get(f\"{API_URL}/health\", timeout=3)\n",
    "    print(\"✅ SentinelZ API Connection Status:\", resp.status_code)\n",
    "    print(\"Health Details:\", resp.json())\n",
    "except Exception as e:\n",
    "    print(\"⚠️ Could not reach API. Please ensure the API is running:\", e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.1: Load Test Stream Transactions\n",
    "data = joblib.load('../data/processed_data.joblib')\n",
    "X_test = data['X_test']\n",
    "y_test = data['y_test']\n",
    "\n",
    "# Sample 300 transactions (including known fraud and legitimate transactions)\n",
    "fraud_indices = y_test[y_test == 1].index\n",
    "legit_indices = y_test[y_test == 0].sample(270, random_state=42).index\n",
    "stream_indices = list(fraud_indices) + list(legit_indices)\n",
    "np.random.seed(42)\n",
    "np.random.shuffle(stream_indices)\n",
    "\n",
    "stream_X = X_test.loc[stream_indices]\n",
    "stream_y = y_test.loc[stream_indices]\n",
    "\n",
    "print(f\"Prepared streaming queue of {len(stream_X)} transactions:\")\n",
    "print(f\"  • Legitimate Transactions : {(stream_y == 0).sum()}\")\n",
    "print(f\"  • Fraudulent Injections   : {(stream_y == 1).sum()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.2: Execute Streaming In-Flight Inference\n",
    "results = []\n",
    "print(\"⚡ Starting real-time transaction streaming simulation on s390x...\")\n",
    "t_stream_start = time.perf_counter()\n",
    "\n",
    "for idx, (original_idx, row) in enumerate(stream_X.iterrows()):\n",
    "    payload = {\n",
    "        \"transaction_id\": f\"TXN-{idx+1:04d}\",\n",
    "        \"features\": row.values.tolist()\n",
    "    }\n",
    "    \n",
    "    t_req = time.perf_counter()\n",
    "    resp = requests.post(f\"{API_URL}/score\", json=payload)\n",
    "    roundtrip_ms = (time.perf_counter() - t_req) * 1000.0\n",
    "    \n",
    "    data_res = resp.json()\n",
    "    data_res['actual_label'] = int(stream_y.loc[original_idx])\n",
    "    data_res['roundtrip_ms'] = round(roundtrip_ms, 3)\n",
    "    results.append(data_res)\n",
    "\n",
    "total_stream_time = time.perf_counter() - t_stream_start\n",
    "df_stream = pd.DataFrame(results)\n",
    "print(f\"\\n✅ Stream simulation completed in {total_stream_time:.2f} seconds!\")\n",
    "print(f\"Total Throughput: {len(df_stream) / total_stream_time:.1f} Transactions/Second (TPS)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.3: Real-Time Stream Performance Summary\n",
    "n_total = len(df_stream)\n",
    "n_blocked = (df_stream['decision'] == 'BLOCK').sum()\n",
    "n_approved = (df_stream['decision'] == 'APPROVE').sum()\n",
    "\n",
    "# Ground truth accuracy on stream\n",
    "true_positives = ((df_stream['decision'] == 'BLOCK') & (df_stream['actual_label'] == 1)).sum()\n",
    "false_positives = ((df_stream['decision'] == 'BLOCK') & (df_stream['actual_label'] == 0)).sum()\n",
    "actual_fraud = (df_stream['actual_label'] == 1).sum()\n",
    "\n",
    "print(\"=\" * 55)\n",
    "print(\"        STREAMING INFERENCE METRICS (s390x)\")\n",
    "print(\"=\" * 55)\n",
    "print(f\"Total Transactions Streamed : {n_total}\")\n",
    "print(f\"Approved Transactions       : {n_approved} ({(n_approved/n_total)*100:.1f}%)\")\n",
    "print(f\"Blocked Transactions        : {n_blocked} ({(n_blocked/n_total)*100:.1f}%)\")\n",
    "print(f\"Fraud Caught (Recall)       : {true_positives} / {actual_fraud} ({(true_positives/actual_fraud)*100:.1f}%)\")\n",
    "print(f\"False Positive Declines     : {false_positives}\")\n",
    "print(\"-\" * 55)\n",
    "print(f\"Mean Inference Latency      : {df_stream['latency_ms'].mean():.3f} ms\")\n",
    "print(f\"P50 (Median) Latency        : {df_stream['latency_ms'].median():.3f} ms\")\n",
    "print(f\"P95 Latency                 : {df_stream['latency_ms'].quantile(0.95):.3f} ms\")\n",
    "print(f\"P99 Tail Latency            : {df_stream['latency_ms'].quantile(0.99):.3f} ms\")\n",
    "print(f\"Mean Roundtrip Latency (API): {df_stream['roundtrip_ms'].mean():.3f} ms\")\n",
    "print(\"=\" * 55)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.4: Streaming Benchmark Visualizations\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Plot 1: In-Flight Latency Distribution with Banking SLA\n",
    "axes[0].plot(df_stream.index, df_stream['latency_ms'], color='#0f62fe', alpha=0.7, label='In-Flight Inference Latency')\n",
    "axes[0].axhline(5.0, color='#da1e28', linestyle='--', linewidth=2, label='Strict Banking SLA Limit (5.0 ms)')\n",
    "axes[0].axhline(df_stream['latency_ms'].mean(), color='#198038', linestyle='-', linewidth=2, label=f\"Mean Latency ({df_stream['latency_ms'].mean():.2f} ms)\")\n",
    "axes[0].set_title('Real-Time Transaction Inference Latency on IBM LinuxONE (s390x)', fontweight='bold')\n",
    "axes[0].set_xlabel('Transaction Sequence')\n",
    "axes[0].set_ylabel('Inference Latency (ms)')\n",
    "axes[0].legend(loc='upper right')\n",
    "axes[0].set_ylim(0, max(6.0, df_stream['latency_ms'].max() * 1.2))\n",
    "\n",
    "# Plot 2: Decision Outcomes Breakdown\n",
    "axes[1].pie(\n",
    "    [n_approved, n_blocked],\n",
    "    labels=['Approved (Legitimate)', 'Blocked (Scam / Fraud)'],\n",
    "    autopct='%1.1f%%',\n",
    "    colors=['#24a148', '#da1e28'],\n",
    "    explode=(0, 0.1),\n",
    "    startangle=140\n",
    ")\n",
    "axes[1].set_title('Pre-Settlement Interception Ratio', fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../reports/streaming_latency_benchmark.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4.5: Save Stream Benchmark Log\n",
    "df_stream.to_csv('../reports/streaming_results.csv', index=False)\n",
    "print(\"✅ Streaming benchmark report exported:\")\n",
    "print(\"  - Chart: ../reports/streaming_latency_benchmark.png\")\n",
    "print(\"  - CSV  : ../reports/streaming_results.csv\")\n",
    "print(\"\\n🎉 PHASE 4 COMPLETE! Ready for Phase 5: SHAP Explainability & Automated GenAI SAR Generation.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("c:\\Users\\kiran\\Desktop\\IBM Z DATATHON\\notebooks\\03_Streaming_Simulation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("03_Streaming_Simulation.ipynb created successfully!")
