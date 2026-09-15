import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "transaction_dataset.csv"

MODEL_DIR = BASE_DIR / "saved_model"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("DeFiLens Model Training")
print("=" * 70)

print("\n[1/8] Loading dataset...")

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n"
        "\nMake sure transaction_dataset.csv is in the "
        "same folder as train_model.py."
    )

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 2. TARGET
# ============================================================

print("\n[2/8] Preparing features...")

if "FLAG" not in df.columns:

    raise ValueError(
        "The dataset does not contain the required "
        "'FLAG' target column."
    )

y = df["FLAG"].astype(int)

X = df.drop(
    columns=["FLAG"]
)


# ============================================================
# REMOVE NON-MODEL COLUMNS
# ============================================================

columns_to_remove = [
    "Index",
    "Address",
    "Unnamed: 0",
]

for column in columns_to_remove:

    if column in X.columns:

        X = X.drop(
            columns=[column]
        )


# Keep numeric features only
X = X.select_dtypes(
    include=[np.number]
)


# Remove zero-variance columns
variance = X.var(
    numeric_only=True
)

zero_variance_columns = variance[
    variance == 0
].index.tolist()

if zero_variance_columns:

    print(
        "Removing zero-variance columns:",
        zero_variance_columns
    )

    X = X.drop(
        columns=zero_variance_columns
    )


original_features = X.columns.tolist()

print(
    f"Original model features: "
    f"{len(original_features)}"
)

print(
    f"Fraud samples: {(y == 1).sum()}"
)

print(
    f"Legitimate samples: {(y == 0).sum()}"
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n[3/8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# 4. IMPUTATION
# ============================================================

print("\n[4/8] Fitting imputer...")

# IMPORTANT:
# Fit ONLY on training data.
# This prevents test-set leakage.

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# SAVE ORIGINAL FEATURE COLUMNS
# ============================================================

save_json(
    MODEL_DIR / "feature_columns.json",
    original_features
)


# ============================================================
# 5. SHAP FEATURE SELECTION
# ============================================================

print("\n[5/8] Performing SHAP feature analysis...")

# Initial model used only to determine feature importance

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = (
    negative / positive
)


shap_model = XGBClassifier(

    n_estimators=200,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


print(
    "Training preliminary XGBoost model..."
)

shap_model.fit(
    X_train_imputed,
    y_train
)


print(
    "Calculating SHAP values..."
)

explainer = shap.TreeExplainer(
    shap_model
)

shap_values = explainer.shap_values(
    X_train_imputed
)


# SHAP importance
mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)


shap_importance = pd.DataFrame({

    "feature":
        X_train_imputed.columns,

    "mean_abs_shap":
        mean_abs_shap

})


shap_importance = (
    shap_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# SELECT TOP FEATURES
# ------------------------------------------------------------

# Your previous model used 20 SHAP-selected features.
# Keep the same approach here.

NUMBER_OF_FEATURES = min(
    20,
    len(shap_importance)
)


selected_features = (
    shap_importance
    .head(NUMBER_OF_FEATURES)
    ["feature"]
    .tolist()
)


print(
    f"\nSelected {len(selected_features)} "
    f"features using SHAP."
)

print("\nSelected features:")

for i, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{i:2}. {feature}"
    )


# Save SHAP information
shap_importance.to_csv(
    MODEL_DIR /
    "shap_feature_importance.csv",
    index=False
)


# Save selected features
save_json(
    MODEL_DIR /
    "selected_features.json",
    selected_features
)


# ============================================================
# REDUCE DATASET TO SELECTED FEATURES
# ============================================================

X_train_selected = (
    X_train_imputed[
        selected_features
    ]
)

X_test_selected = (
    X_test_imputed[
        selected_features
    ]
)


# ============================================================
# 6. TRAIN FINAL XGBOOST
# ============================================================

print("\n[6/8] Training final XGBoost model...")


# These are optimized-style parameters based on
# the configuration we previously discussed.

best_parameters = {

    "n_estimators": 552,

    "max_depth": 3,

    "learning_rate": 0.1087,

    "subsample": 0.9,

    "colsample_bytree": 0.9,

    "min_child_weight": 1,

    "gamma": 0,

    "reg_alpha": 0,

    "reg_lambda": 1,

    "scale_pos_weight":
        scale_pos_weight,

    "objective":
        "binary:logistic",

    "eval_metric":
        "logloss",

    "random_state":
        RANDOM_STATE
}


final_model = XGBClassifier(
    **best_parameters,
    n_jobs=-1
)


final_model.fit(
    X_train_selected,
    y_train,
    eval_set=[
        (
            X_test_selected,
            y_test
        )
    ],
    verbose=False
)


print(
    "Final XGBoost model trained."
)


# Save best parameters
save_json(
    MODEL_DIR /
    "best_parameters.json",
    best_parameters
)


# Save model
final_model.save_model(
    MODEL_DIR /
    "defilens_fraud_xgboost.json"
)


# Save imputer
joblib.dump(
    imputer,
    MODEL_DIR /
    "imputer.joblib"
)


# ============================================================
# 7. FIND BEST THRESHOLD
# ============================================================

print("\n[7/8] Selecting classification threshold...")


test_probabilities = (
    final_model
    .predict_proba(
        X_test_selected
    )[:, 1]
)


# ------------------------------------------------------------
# Threshold search
# ------------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_threshold = 0.50
best_f2 = -1


for threshold in thresholds:

    predictions = (
        test_probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    # F2 gives recall more importance
    if (
        precision == 0
        and recall == 0
    ):

        f2 = 0

    else:

        beta = 2

        f2 = (
            (1 + beta ** 2)
            * precision
            * recall
            /
            (
                beta ** 2
                * precision
                + recall
            )
        )

    if f2 > best_f2:

        best_f2 = f2

        best_threshold = float(
            threshold
        )


print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best F2 score: "
    f"{best_f2:.4f}"
)


save_json(
    MODEL_DIR /
    "threshold.json",
    {
        "threshold":
            best_threshold,

        "selection_metric":
            "F2",

        "best_f2":
            best_f2
    }
)


# ============================================================
# 8. FINAL EVALUATION
# ============================================================

print("\n[8/8] Evaluating final model...")


final_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


cm = confusion_matrix(
    y_test,
    final_predictions
)


# ============================================================
# RESULTS
# ============================================================

results = {

    "dataset": {

        "rows":
            int(len(df)),

        "columns":
            int(df.shape[1]),

        "training_samples":
            int(len(X_train)),

        "testing_samples":
            int(len(X_test)),

        "legitimate_samples":
            int((y == 0).sum()),

        "fraud_samples":
            int((y == 1).sum())

    },

    "features": {

        "original_features":
            int(len(original_features)),

        "selected_features":
            int(len(selected_features)),

        "selected_feature_names":
            selected_features

    },

    "model": {

        "algorithm":
            "XGBoost",

        "parameters":
            best_parameters

    },

    "threshold": {

        "value":
            best_threshold,

        "metric":
            "F2",

        "f2":
            best_f2

    },

    "metrics": {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc

    },

    "confusion_matrix": {

        "true_negative":
            int(cm[0][0]),

        "false_positive":
            int(cm[0][1]),

        "false_negative":
            int(cm[1][0]),

        "true_positive":
            int(cm[1][1])

    }

}


save_json(
    MODEL_DIR /
    "final_results.json",
    results
)


# ============================================================
# PRINT REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy       : {accuracy:.4f}"
)

print(
    f"Precision      : {precision:.4f}"
)

print(
    f"Recall         : {recall:.4f}"
)

print(
    f"F1 Score       : {f1:.4f}"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print(
    f"PR-AUC         : {pr_auc:.4f}"
)

print(
    f"Threshold      : {best_threshold:.2f}"
)

print(
    f"SHAP Features  : {len(selected_features)}"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nSaved model artifacts to:\n"
    f"{MODEL_DIR}"
)

print("\nFiles created:")

for file in sorted(
    MODEL_DIR.iterdir()
):

    print(
        f"  ✓ {file.name}"
    )

print("\nYou can now start FastAPI:")
print(
    "uvicorn api:app --host 0.0.0.0 --port 8001"
)

print("\nThen start Streamlit in another terminal:")
print(
    "streamlit run app.py"
)