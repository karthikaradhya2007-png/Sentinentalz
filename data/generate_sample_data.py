"""
SentinelZ Quick Synthetic Benchmark Generator
Generates a realistic creditcard.csv benchmark sample (Time, V1..V28, Amount, Class)
Use this for instant local testing or while the 150MB Kaggle dataset uploads.
"""
import numpy as np
import pandas as pd
import os

def generate_benchmark_data(n_samples=25000, fraud_ratio=0.003, output_path="data/creditcard.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    np.random.seed(42)
    # Legitimate transactions
    legit_pca = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    legit_time = np.sort(np.random.uniform(0, 172800, size=n_legit))
    legit_amount = np.random.exponential(scale=88.0, size=n_legit)
    legit_class = np.zeros(n_legit, dtype=int)
    
    # Fraud transactions (shifted distribution simulating anomalies)
    fraud_pca = np.random.normal(loc=1.5, scale=2.5, size=(n_fraud, 28))
    fraud_time = np.random.uniform(0, 172800, size=n_fraud)
    fraud_amount = np.random.exponential(scale=122.0, size=n_fraud)
    fraud_class = np.ones(n_fraud, dtype=int)
    
    # Combine
    pca = np.vstack([legit_pca, fraud_pca])
    time_col = np.concatenate([legit_time, fraud_time])
    amount_col = np.concatenate([legit_amount, fraud_amount])
    class_col = np.concatenate([legit_class, fraud_class])
    
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.hstack([time_col.reshape(-1, 1), pca, amount_col.reshape(-1, 1), class_col.reshape(-1, 1)])
    
    df = pd.DataFrame(data, columns=cols)
    df['Class'] = df['Class'].astype(int)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} transactions -> {output_path} (Fraud count: {df['Class'].sum()}, Fraud rate: {df['Class'].mean()*100:.3f}%)")

if __name__ == "__main__":
    generate_benchmark_data()
