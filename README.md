# Simple Single Layer Perceptron

Implementation from scratch of a one-hidden-layer neural network with a scikit-learn-style API. Supports regression and classification (binary and multi-class) using only NumPy.

## Project structure

├── activations.py          # ReLU, tanh, logistic, softmax + their derivatives
├── slp_base.py             # Abstract base class for both estimators
├── slp_classifier.py       # SimpleSLPClassifier (binary + multi-class)
├── slp_regressor.py        # SimpleSLPRegressor
├── tests/                  # pytest unit tests (54 tests)
│   ├── test_activations.py
│   ├── test_models.py
│   └── test_smoke.py
├── report.ipynb            # Jupyter notebook with the four demonstrations
├── notes_revision.md       # Mathematical reasoning behind each design choice
├── requirements.txt        # Python dependencies
└── pytest.ini              # pytest configuration

## Installation

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
pip install matplotlib jupyter pandas
```

## Running the tests

All 54 tests should pass:

```bash
pytest -v
```

## Running the report

Open `report.ipynb` in VS Code or Jupyter Lab and run all cells. The notebook demonstrates the implementation on four datasets:

1. **Diabetes** (sklearn) — regression
2. **Breast cancer** (sklearn) — binary classification
3. **Digits** (sklearn) — multi-class classification (bonus)
4. **Wine quality** (Kaggle, `data/winequality-red.csv`) — regression on a real-world noisy dataset

Each demonstration compares the implementation against the equivalent `sklearn.MLPRegressor` or `MLPClassifier`.

## Quick API example

```python
from slp_classifier import SimpleSLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = SimpleSLPClassifier(hidden_layer_size=30, max_iter=300, random_state=42)
clf.fit(X_train, y_train)

print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
```

## Design choices

A few decisions deviate from the scaffold provided in the brief. Each is justified in `notes_revision.md`.

- Default activation is `"relu"` rather than `"logistic"`: avoids saturation and gradient explosion on regression targets with large magnitudes.
- Default learning rate is `0.01` rather than `0.001`: calibrated for vanilla gradient descent rather than adam.
- Weight initialisation uses σ = 0.15 (close to He initialisation) rather than σ = 0.01: prevents dying ReLU on small networks.
- Softmax + cross-entropy is used for both binary and multi-class classification (same code path).
- Target normalisation is applied in regression demos with large-magnitude targets (recommended in the brief).

## Notes for the defence

The mathematical derivations, the gradient explosion incident on diabetes, the bias-variance trade-off observed on wine quality, and the implicit regularisation property of vanilla SGD are all documented in `notes_revision.md`.