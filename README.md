# 🧪 Modular ADMET Predictive Pipeline

An object-oriented, production-ready machine learning pipeline designed to predict pharmacokinetic (ADMET) properties of small molecules. Built with a focus on scalable software architecture, mathematical rigor, and reproducible research.

## 🚀 Core Architecture

* **Decoupled Configuration:** A YAML-driven architecture (`endpoints.yaml`) that completely separates model parameters, feature selection, and data routing from the underlying Python source code.
* **High-Resolution Feature Engineering:** Extracts 2048-bit Morgan Fingerprints combined with global 2D RDKit descriptors, drastically reducing structural hash collisions compared to standard 1024-bit vectors.
* **Bayesian Hyperparameter Optimization:** Integrates `Optuna` to dynamically tune models prior to training. Utilizes mathematically strict L1 (Lasso) and L2 (Ridge) regularization to compress noisy feature weights to zero, effectively rescuing regression tasks from the curse of dimensionality.
* **Rigorous Chemical Generalization:** Employs **Scaffold Splitting** (via PyTDC) instead of random splits to ensure the models learn foundational chemical rules rather than memorizing overlapping molecular backbones.

## 📂 Repository Structure

```text
├── configs/
│   └── endpoints.yaml
├── data/                  # Ignored in version control
├── notebooks/             
├── experiments/
├── outputs/
│   ├── models/            # Serialized .pkl files
│   ├── plots/
│   └── results/           # Exported CSV leaderboards
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split.py
│   ├── features.py
│   ├── inference.py
│   ├── evaluate.py
│   └── models/
│       ├── __init__.py
│       ├── model_factory.py
│       └── optimize.py
├── main.py
├── README.md
└── requirements.txt