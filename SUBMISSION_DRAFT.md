# SentinelZ — Real-Time In-Flight Fraud Prevention on IBM LinuxONE

## IBM Z Datathon 2026 Submission Document

---

### 1. Problem Statement
**Global financial fraud extracts over $485 billion annually**, driven by high-velocity Authorized Push Payment (APP) scams, account takeovers, and card-not-present (CNP) fraud. Traditional enterprise banking systems still rely on **asynchronous, post-transaction batch analytics (T+1 settlement)** or slow off-host third-party APIs that introduce 50–100ms of latency—exceeding core banking transaction SLAs. Consequently, fraudulent transfers settle before detection models can trigger an alert, leaving financial institutions and victims unable to recover stolen funds. Conversely, crude static rule engines create high false-positive rates (often exceeding 80–90%), arbitrarily locking accounts and damaging consumer trust. 

There is an urgent need for an **in-flight, sub-millisecond AI scoring engine** co-located directly on mission-critical transaction infrastructure (IBM LinuxONE) that can evaluate, score, and block illicit transactions in real time before funds clear.

---

### 2. Proposed Solution (Word Count: 154 words — Limit: < 200 words)
**SentinelZ** is a high-throughput, in-flight fraud prevention engine architected natively for **IBM LinuxONE** to detect and intercept financial crime in real time before settlement. Operating directly alongside core banking transaction streams, SentinelZ eliminates the high-latency network round-trips of off-host fraud scoring, achieving sub-millisecond inference speeds.

The system implements a dual-tier **Hybrid AI architecture** aligned with IBM Z capabilities:
1. **In-Flight Predictive AI:** Ultra-low-latency quantized gradient-boosted trees and neural models running on LinuxONE to instantaneously score transactions and block illicit transfers within strict banking SLAs (<10ms).
2. **Asynchronous Generative AI & Explainability:** Post-scoring SHAP interpretability coupled with lightweight LLMs to automatically synthesize human-readable Suspicious Activity Reports (SAR) and regulatory audit trails.

By deploying via containerized microservices on LinuxONE Community Cloud, SentinelZ leverages s390x high-concurrency processing to process thousands of transactions per second. This ensures ironclad protection against fraud while slashing false-positive declines that frustrate legitimate consumers.

---

### 3. Target Audience

1. **Commercial Banks & Payment Service Providers (PSPs):**
   * *Stakeholders:* Chief Risk Officers (CROs), Fraud Operations Directors, and Real-Time Settlement Engineering teams.
   * *Impact:* Need sub-millisecond transaction inference to intercept fraudulent transfers pre-settlement, minimize catastrophic chargebacks, and slash false-positive account freezes to protect interchange revenue.

2. **Financial Regulators & Compliance Officers:**
   * *Stakeholders:* Anti-Money Laundering (AML) investigators, Financial Crimes Enforcement Network (FinCEN) auditors, and Central Bank compliance directors.
   * *Impact:* Require transparent, explainable AI decisions with immutable, automated audit trails and Suspicious Activity Report (SAR) generation to satisfy stringent governance and fair-lending requirements.

3. **Consumers, Vulnerable Citizens & Merchants:**
   * *Stakeholders:* Retail bank customers, small business owners, and digitally vulnerable populations (such as the elderly targeted by social engineering scams).
   * *Impact:* Receive proactive, zero-latency protection against account takeovers and payment fraud without suffering humiliating, unwarranted card declines or account lockouts during legitimate purchases.

---

### 4. Datasets Selected

1. **IEEE-CIS Fraud Detection Benchmark Dataset:**
   * *Characteristics:* 590,000+ real-world e-commerce transactions across hundreds of anonymized features including device information, card metadata, billing geolocation, and transaction frequency velocity.
   * *Role in Project:* Trains the primary multi-feature predictive model to evaluate complex behavioral patterns and cross-feature interactions at enterprise scale.

2. **Credit Card Fraud Detection Dataset (ULB / Kaggle):**
   * *Characteristics:* 284,807 European cardholder transactions transformed via Principal Component Analysis (PCA) with an extreme class imbalance (0.172% fraudulent transactions).
   * *Role in Project:* Serves as the high-precision stress-test benchmark to evaluate model sensitivity, Area Under Precision-Recall Curve (AUPRC), and cost-sensitive classification under severe data skew.

3. **PaySim Synthetic Financial Simulator Dataset:**
   * *Characteristics:* Multi-million synthetic transaction log generated from real financial logs, modeling mobile money transfers (`CASH_OUT`, `TRANSFER`, `PAYMENT`) and simulating account-draining scam behaviors.
   * *Role in Project:* Deployed inside the LinuxONE Community Cloud JupyterLab environment as a mock real-time message stream to simulate high-concurrency streaming inference, evaluating throughput (transactions/sec) and sub-10ms response latencies under heavy load.

---

### 5. Architectural Alignment with IBM Z / LinuxONE

* **Theme 1 Alignment:** *Real-Time AI for Critical Decisions (Detect. Decide. Respond.)* — directly matches the #1 theme focus area (Fraud Detection).
* **Mainframe Hardware Affinity:** Over 70% of global transaction value runs on IBM Z mainframes. Co-locating AI inference with the transactional ledger eliminates external network hops, preserving transaction throughput and sub-millisecond latencies.
* **Hybrid AI Framework:** Direct fulfillment of both IBM Z Datathon mandatory AI labs:
  * *Lab 1:* Multi-model predictive ensemble for high-speed scoring.
  * *Lab 2:* Predictive AI + Generative AI for automated narrative SAR generation and explainability.
