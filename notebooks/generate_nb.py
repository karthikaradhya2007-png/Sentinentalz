import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🛡️ SentinelZ: Phase 1 — Exploratory Data Analysis (EDA) & Preprocessing\n",
    "**IBM Z Datathon 2026 | Theme 1: Real-Time AI for Critical Decisions**  \n",
    "*Platform:* **IBM LinuxONE Community Cloud (s390x)**\n",
    "\n",
    "---\n",
    "\n",
    "### Objectives for Phase 1:\n",
    "1. **Architecture Verification:** Confirm execution on `s390x` architecture.\n",
    "2. **Dataset Inspection:** Load ULB Credit Card Fraud dataset and inspect schema & data skew.\n",
    "3. **Statistical Analysis & Visualization:** Analyze extreme class imbalance and feature behaviors.\n",
    "4. **Feature Engineering & Scaling:** Normalize transaction `Amount` and `Time` attributes.\n",
    "5. **Imbalance Resampling (SMOTE):** Balance the training distribution using Synthetic Minority Over-sampling Technique.\n",
    "6. **Export Artifacts:** Persist preprocessed train/test partitions and scalers for Phase 2 multi-model ensemble training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.0: Verify s390x Architecture\n",
    "import platform, sys\n",
    "print(f\"Machine Architecture : {platform.machine()}\")\n",
    "print(f\"Processor            : {platform.processor()}\")\n",
    "print(f\"Python Version       : {sys.version}\")\n",
    "assert platform.machine() == 's390x', \"Warning: Not running on IBM LinuxONE s390x!\"\n",
    "print(\"✅ Confirmed execution on IBM LinuxONE (s390x)!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.1: Import Core Libraries\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "# Set visual style\n",
    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
    "os.makedirs('../reports', exist_ok=True)\n",
    "os.makedirs('../models', exist_ok=True)\n",
    "print(\"✅ Libraries and directories successfully initialized.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.2: Load Dataset\n",
    "data_path = '../data/creditcard.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "\n",
    "print(f\"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\")\n",
    "print(f\"Missing Values: {df.isnull().sum().max()}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.3: Class Imbalance Analysis\n",
    "counts = df['Class'].value_counts()\n",
    "legit_count = counts.get(0, 0)\n",
    "fraud_count = counts.get(1, 0)\n",
    "fraud_pct = (fraud_count / len(df)) * 100\n",
    "\n",
    "print(\"=\" * 45)\n",
    "print(\"        CLASS DISTRIBUTION SUMMARY\")\n",
    "print(\"=\" * 45)\n",
    "print(f\"Legitimate Transactions (0): {legit_count:,} ({100 - fraud_pct:.3f}%)\")\n",
    "print(f\"Fraudulent Transactions (1): {fraud_count:,} ({fraud_pct:.3f}%)\")\n",
    "print(f\"Imbalance Ratio            : 1 : {int(legit_count / max(1, fraud_count))}\")\n",
    "print(\"=\" * 45)\n",
    "\n",
    "print(\"\\n--- Transaction Amount Statistics by Class ---\")\n",
    "print(df.groupby('Class')['Amount'].describe().round(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.4: Visualizing Imbalance & Amount Distribution\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Plot 1: Class Distribution\n",
    "sns.barplot(x=['Legitimate (0)', 'Fraudulent (1)'], y=[legit_count, fraud_count], ax=axes[0], palette=['#2b5c8f', '#d9534f'])\n",
    "axes[0].set_yscale('log')\n",
    "axes[0].set_title('Class Distribution (Log Scale) — SentinelZ on LinuxONE', fontsize=12, fontweight='bold')\n",
    "axes[0].set_ylabel('Number of Transactions (Log)')\n",
    "for p in axes[0].patches:\n",
    "    axes[0].annotate(f\"{int(p.get_height()):,}\", (p.get_x() + p.get_width() / 2., p.get_height()),\n",
    "                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')\n",
    "\n",
    "# Plot 2: Transaction Amounts Boxplot\n",
    "sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1], palette=['#2b5c8f', '#d9534f'], showfliers=False)\n",
    "axes[1].set_xticklabels(['Legit (0)', 'Fraud (1)'])\n",
    "axes[1].set_title('Transaction Amount Spread (Excl. Extreme Outliers)', fontsize=12, fontweight='bold')\n",
    "axes[1].set_ylabel('Amount ($)')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('../reports/class_imbalance.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.5: Feature Correlation with Fraud Label\n",
    "plt.figure(figsize=(12, 6))\n",
    "correlations = df.corr()['Class'].drop('Class').sort_values()\n",
    "top_corr = pd.concat([correlations.head(7), correlations.tail(7)])\n",
    "\n",
    "colors = ['#d9534f' if x > 0 else '#2b5c8f' for x in top_corr.values]\n",
    "sns.barplot(x=top_corr.values, y=top_corr.index, palette=colors)\n",
    "plt.title('Top Negative & Positive Feature Correlations with Fraud (Class)', fontsize=13, fontweight='bold')\n",
    "plt.xlabel('Pearson Correlation Coefficient')\n",
    "plt.axvline(0, color='gray', linestyle='--', alpha=0.7)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../reports/feature_correlations.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.6: Feature Preprocessing & Scaling\n",
    "df_clean = df.copy()\n",
    "\n",
    "# V1 to V28 are already PCA transformed; scale Amount and Time\n",
    "scaler = StandardScaler()\n",
    "df_clean['scaled_Amount'] = scaler.fit_transform(df_clean[['Amount']])\n",
    "df_clean['scaled_Time'] = scaler.fit_transform(df_clean[['Time']])\n",
    "\n",
    "# Drop unscaled raw features\n",
    "df_clean.drop(['Amount', 'Time'], axis=1, inplace=True)\n",
    "\n",
    "# Partition into Feature Matrix X and Target y\n",
    "X = df_clean.drop('Class', axis=1)\n",
    "y = df_clean['Class']\n",
    "\n",
    "print(f\"Prepared Feature Matrix X : {X.shape}\")\n",
    "print(f\"Target Vector y          : {y.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.7: Stratified Train-Test Split (80/20)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "print(f\"Training set samples : {len(X_train):,} (Fraud: {y_train.sum():,})\")\n",
    "print(f\"Testing set samples  : {len(X_test):,} (Fraud: {y_test.sum():,})\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.8: Apply SMOTE on Training Partition (Class Balancing)\n",
    "print(\"Applying SMOTE to balance the training set on s390x...\")\n",
    "smote = SMOTE(random_state=42, sampling_strategy='auto')\n",
    "X_train_res, y_train_res = smote.fit_resample(X_train, y_train)\n",
    "\n",
    "print(\"=\" * 45)\n",
    "print(\"        AFTER SMOTE RESAMPLING\")\n",
    "print(\"=\" * 45)\n",
    "print(f\"Legitimate Samples in Train: {(y_train_res == 0).sum():,}\")\n",
    "print(f\"Fraudulent Samples in Train: {(y_train_res == 1).sum():,}\")\n",
    "print(f\"New Resampled Train Shape  : {X_train_res.shape}\")\n",
    "print(\"=\" * 45)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1.9: Save Processed Partitions & Scaler for Phase 2\n",
    "joblib.dump(scaler, '../models/scaler.pkl')\n",
    "\n",
    "processed_data = {\n",
    "    'X_train_res': X_train_res,\n",
    "    'y_train_res': y_train_res,\n",
    "    'X_test': X_test,\n",
    "    'y_test': y_test,\n",
    "    'feature_names': list(X.columns)\n",
    "}\n",
    "joblib.dump(processed_data, '../data/processed_data.joblib')\n",
    "\n",
    "print(\"✅ Phase 1 Artifacts successfully saved:\")\n",
    "print(\"  - Scaler       : ../models/scaler.pkl\")\n",
    "print(\"  - Train/Test   : ../data/processed_data.joblib\")\n",
    "print(\"  - Reports      : ../reports/class_imbalance.png, ../reports/feature_correlations.png\")\n",
    "print(\"\\n🚀 PHASE 1 COMPLETE! Ready for Phase 2: Hybrid AI Model Training.\")"
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

with open("c:\\Users\\kiran\\Desktop\\IBM Z DATATHON\\notebooks\\01_EDA_and_Preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Notebook created successfully!")
