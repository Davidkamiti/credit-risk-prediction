"""Rebuild the credit-risk model from ml_data.csv and save it for the Streamlit app.

Why this script exists
----------------------
The notebook saved only the Random Forest, but the model was trained on
*scaled* data. The app needs the scaler too, so this script reproduces the
notebook steps (same split, SMOTE, scaler and tuned parameters) and saves
scaler + model together as one Pipeline.

Run it once, from the repo folder:
    pip install pandas scikit-learn imbalanced-learn joblib
    python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "ml_data.csv"
MODEL_PATH = Path("model/credit_risk_model.joblib")
THRESHOLD = 0.6  # the threshold used in the notebook's final evaluation

# Best parameters found by GridSearchCV in Modeling.ipynb
BEST_PARAMS = dict(
    n_estimators=300,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Same split and oversampling as the notebook
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Scaler + model in one object, so the app can never forget to scale
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**BEST_PARAMS)),
    ]
)
model.fit(X_train_smote, y_train_smote)

# Evaluate on the untouched test set so the README numbers can be verified
proba = model.predict_proba(X_test)[:, 1]
print("=== Default threshold (0.5) ===")
print(classification_report(y_test, model.predict(X_test)))
print(f"=== Threshold {THRESHOLD} ===")
pred = (proba >= THRESHOLD).astype(int)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

# Save (compressed: a 300-tree forest can be too large for GitHub otherwise)
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH, compress=3)
size_mb = MODEL_PATH.stat().st_size / 1e6
print(f"\nSaved {MODEL_PATH} ({size_mb:.1f} MB)")
if size_mb > 90:
    print(
        "WARNING: GitHub rejects files over 100 MB. Re-run with n_estimators=100 "
        "and min_samples_leaf=2 in BEST_PARAMS to shrink the model."
    )

# Record the settings the app should use, and the library version it must match
Path("model/model_info.json").write_text(
    json.dumps({"threshold": THRESHOLD, "sklearn_version": sklearn.__version__}, indent=2)
)
Path("requirements.txt").write_text(
    f"streamlit\npandas\nnumpy\njoblib\nscikit-learn=={sklearn.__version__}\n"
)
print("Wrote model/model_info.json and requirements.txt")
