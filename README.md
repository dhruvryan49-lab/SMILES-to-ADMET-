# ADMET-GATv2: Hybrid Graph Attention Networks for Molecular Affinity

A complete, end-to-end machine learning pipeline for molecular property prediction and drug-target affinity modeling. This project evolves from classical machine learning baselines (XGBoost) into a custom PyTorch Geometric deep learning architecture, combining multi-head graph attention with deterministic thermodynamic descriptors.

## 🏗️ Architecture: The Hybrid GATv2
Standard Graph Neural Networks struggle with global molecular properties because they only observe local bond topographies. This architecture solves that via a dual-branch fusion system:
1. **Spatial Branch:** A 3-layer Graph Attention Network (`GATv2Conv`) processes 3D molecular graphs, calculating attention coefficients for atom-bond interactions.
2. **Physics Branch:** RDKit computes 8 deterministic global descriptors (LogP, TPSA, Molecular Weight, etc.).
3. **Fusion:** The 64D global-pooled graph tensor is concatenated with the physics tensor prior to the final prediction layers.

## 🗄️ Project Structure
```text
KBG/
├── configs/               # YAML configurations for PyTDC endpoints
├── data/                  # Raw and processed datasets (ChEMBL, PyTDC, MUV)
├── outputs/               # Model weights (.pt, .pkl), logs, and plots
├── scripts/               # Data mining utilities (fetch_chembl.py)
├── src/                   # Core modules
│   ├── models/            # Neural network architectures and trainers (gnn.py, gnn_trainer.py, etc.)
│   ├── data_loader.py     # PyG DataLoaders and batch processing
│   ├── graph_features.py  # 3D tensor and thermodynamic descriptor extraction
│   └── preprocessing.py   # Data cleaning and log transformations
├── main_chembl.py         # ChEMBL EGFR optimization pipeline
├── main_gnn.py            # Vanilla GNN baseline
├── main_muv.py            # MUV multi-task limit test
├── optimize_chembl.py     # Optuna Bayesian hyperparameter search
└── requirements.txt       # Environment dependencies
```

## 🧪 Data Engineering & Benchmarking
- **Target Affinity (ChEMBL API):** Live data mining script targeting the EGFR protein. Aggregates multi-lab bioassay data, cleans duplicates, and performs logarithmic transformations into mathematically stable $pIC_{50}$ concentrations.
- **ADMET Benchmarks (PyTDC):** Automated multi-target evaluation across systemic bodily functions (BBB penetration, hERG toxicity, Half-life).
- **Stress Testing (MoleculeNet / MUV):** Multi-task learning pipeline designed to test the model against maximum unbiased validation sets containing extreme class imbalance.

## 📊 Key Results & Optimization
The architecture was optimized using **Optuna** for Bayesian hyperparameter searching (learning rate, dropout, hidden channels) alongside Early Stopping to prevent overfitting.

* **EGFR Affinity (ChEMBL):** Achieved an $R^2$ of **0.457** and RMSE of **1.003** on unseen test data, successfully mapping underlying binding thermodynamics.
* **MUV Stress Test:** Demonstrated the limits of training from scratch. Identified gradient collapse (Loss $\approx 0.693$, representing $-\ln(0.5)$) caused by MUV's aggressive stripping of scaffold bias and extreme active/inactive class imbalance, highlighting the necessity of self-supervised pre-training for heavily debiased datasets.

## 🚀 Usage

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Pipelines**
```bash
# Run the PyTDC Multi-Endpoint Benchmark
python main.py

# Run the Optuna Hyperparameter Search
python optimize_chembl.py

# Run the optimized Hybrid GATv2 on ChEMBL EGFR data
python main_chembl.py

# Run the MUV Multi-Task Stress Test
python main_muv.py
```