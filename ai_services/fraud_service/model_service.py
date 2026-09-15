import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, DMatrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

DATASET_PATH = BASE_DIR / "transaction_dataset.csv"

MODEL_PATH = MODEL_DIR / "defilens_fraud_xgboost.json"
IMPUTER_PATH = MODEL_DIR / "imputer.joblib"
SELECTED_FEATURES_PATH = MODEL_DIR / "selected_features.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
SHAP_PATH = MODEL_DIR / "shap_feature_importance.csv"
RESULTS_PATH = MODEL_DIR / "final_results.json"


class DeFiLensModel:

    def __init__(self):
        self.model = None
        self.imputer = None
        self.feature_columns = []
        self.selected_features = []
        self.threshold = 0.5
        self.shap_features = None
        self.results = {}
        self.dataset = None
        self.address_column = None
        self.load_all()

    # ========================================================
    # LOAD
    # ========================================================

    def load_all(self):
        self.load_model()
        self.load_imputer()
        self.load_feature_columns()
        self.load_features()
        self.load_threshold()
        self.load_shap()
        self.load_results()
        self.load_dataset()

    def load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found:\n{MODEL_PATH}")

        self.model = XGBClassifier()
        self.model.load_model(str(MODEL_PATH))
        print("✓ XGBoost model loaded")

    def load_imputer(self):
        if not IMPUTER_PATH.exists():
            raise FileNotFoundError(f"Imputer not found:\n{IMPUTER_PATH}")

        self.imputer = joblib.load(IMPUTER_PATH)
        print("✓ Imputer loaded")

    def load_feature_columns(self):
        if hasattr(self.imputer, "feature_names_in_"):
            self.feature_columns = self.imputer.feature_names_in_.tolist()
            print(
                f"✓ Loaded {len(self.feature_columns)} "
                "original features from imputer"
            )
            return

        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                self.feature_columns = data.get(
                    "feature_columns",
                    data.get("features", [])
                )
            else:
                self.feature_columns = data

            print(
                f"✓ Loaded {len(self.feature_columns)} "
                "original features from JSON"
            )
            return

        raise FileNotFoundError(
            "Could not determine the original training feature columns."
        )

    def load_features(self):
        if not SELECTED_FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Selected feature file not found:\n{SELECTED_FEATURES_PATH}"
            )

        with open(
            SELECTED_FEATURES_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            self.selected_features = data.get(
                "selected_features",
                data.get("features", [])
            )
        else:
            self.selected_features = data

        print(
            f"✓ Loaded {len(self.selected_features)} "
            "SHAP-selected features"
        )

    def load_threshold(self):
        if not THRESHOLD_PATH.exists():
            self.threshold = 0.5
            print("⚠ Threshold file not found. Using 0.50")
            return

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            self.threshold = float(
                data.get(
                    "threshold",
                    data.get("best_threshold", 0.5)
                )
            )
        else:
            self.threshold = float(data)

        print(f"✓ Threshold loaded: {self.threshold:.4f}")

    def load_shap(self):
        if not SHAP_PATH.exists():
            print("⚠ SHAP importance file not found")
            self.shap_features = None
            return

        try:
            self.shap_features = pd.read_csv(SHAP_PATH)
            print("✓ SHAP feature importance loaded")
        except Exception as e:
            print(f"⚠ Could not load SHAP data: {e}")
            self.shap_features = None

    def load_results(self):
        if not RESULTS_PATH.exists():
            self.results = {}
            return

        try:
            with open(
                RESULTS_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                self.results = json.load(file)
        except Exception as e:
            print(f"⚠ Could not load results: {e}")
            self.results = {}

    def load_dataset(self):
        if not DATASET_PATH.exists():
            print(f"⚠ Dataset not found: {DATASET_PATH}")
            return

        try:
            self.dataset = pd.read_csv(DATASET_PATH)

            for column in ["Address", "address", "wallet_address", "borrower_address"]:
                if column in self.dataset.columns:
                    self.address_column = column
                    break

            if self.address_column:
                self.dataset["_address_normalized"] = (
                    self.dataset[self.address_column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                print(
                    f"✓ Dataset loaded: {len(self.dataset)} rows"
                )
                print(
                    f"✓ Wallet address column: {self.address_column}"
                )
            else:
                print("⚠ No wallet address column found in dataset")

        except Exception as e:
            print(f"⚠ Could not load dataset: {e}")
            self.dataset = None

    # ========================================================
    # WALLET LOOKUP
    # ========================================================

    @staticmethod
    def normalize_address(address: Optional[str]) -> str:
        if address is None:
            return ""
        return str(address).strip().lower()

    def find_wallet(self, borrower_address: str) -> Dict[str, Any]:
        if self.dataset is None:
            raise ValueError("Transaction dataset is not loaded.")

        if not self.address_column:
            raise ValueError(
                "The transaction dataset does not contain an address column."
            )

        normalized = self.normalize_address(borrower_address)

        if not normalized:
            raise ValueError("Borrower address cannot be empty.")

        matches = self.dataset[
            self.dataset["_address_normalized"] == normalized
        ]

        if matches.empty:
            return {
                "found": False,
                "borrower_address": borrower_address,
                "match_count": 0,
                "rows": []
            }

        # A wallet can occur more than once in transaction datasets.
        # Use the first matching transaction as the model input while
        # reporting all matching rows to the UI.
        rows = matches.drop(
            columns=["_address_normalized"],
            errors="ignore"
        )

        first_row = rows.iloc[0].to_dict()

        return {
            "found": True,
            "borrower_address": borrower_address,
            "matched_address": str(
                first_row.get(self.address_column, borrower_address)
            ),
            "match_count": int(len(rows)),
            "dataset_index": int(rows.index[0]),
            "transaction": {
                k: self.safe_value(v)
                for k, v in first_row.items()
            }
        }

    @staticmethod
    def safe_value(value):
        if value is None:
            return None

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            return value if np.isfinite(value) else None

        if isinstance(value, float):
            return value if np.isfinite(value) else None

        if isinstance(value, np.bool_):
            return bool(value)

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        return value

    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def prepare_features(self, transaction):
        if not transaction:
            raise ValueError("Transaction data is empty.")

        df = pd.DataFrame([transaction])

        df = df.drop(
            columns=[
                "FLAG",
                "Index",
                "Address",
                "address",
                "wallet_address",
                "borrower_address",
                "transaction_hash",
                "Unnamed: 0",
                "_address_normalized"
            ],
            errors="ignore"
        )

        for column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        if hasattr(self.imputer, "feature_names_in_"):
            expected_features = (
                self.imputer.feature_names_in_.tolist()
            )
        else:
            expected_features = self.feature_columns

        if not expected_features:
            raise ValueError("No original training features found.")

        # Reconstruct exact 38-feature training space.
        df = df.reindex(columns=expected_features)

        print(
            f"Incoming transaction fields: {len(transaction)}"
        )
        print(
            f"Features expected by imputer: {len(expected_features)}"
        )
        print(
            f"Features passed to imputer: {len(df.columns)}"
        )

        if list(df.columns) != list(expected_features):
            raise ValueError(
                "Feature names/order do not match the imputer training features."
            )

        transformed = self.imputer.transform(df)

        df = pd.DataFrame(
            transformed,
            columns=expected_features
        )

        if not self.selected_features:
            raise ValueError("No SHAP-selected features loaded.")

        missing_selected = [
            feature for feature in self.selected_features
            if feature not in df.columns
        ]

        if missing_selected:
            raise ValueError(
                "SHAP-selected features missing after imputation:\n"
                + "\n".join(missing_selected)
            )

        df = df[self.selected_features]

        print(
            f"Features passed to XGBoost: {len(df.columns)}"
        )

        return df

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, transaction):
        X = self.prepare_features(transaction)

        probability = float(
            self.model.predict_proba(X)[0][1]
        )

        prediction = int(
            probability >= self.threshold
        )

        return {
            "fraud_probability": probability,
            "prediction": prediction,
            "threshold": self.threshold,
            "features_used": len(self.selected_features)
        }

    # ========================================================
    # BOOSTING PROGRESSION
    # ========================================================

    def get_boosting_progression(self, transaction):
        X = self.prepare_features(transaction)

        booster = self.model.get_booster()
        rounds = booster.num_boosted_rounds()

        if rounds <= 0:
            return []

        # Keep API response reasonably small.
        points = min(rounds, 100)
        iterations = np.unique(
            np.linspace(1, rounds, points, dtype=int)
        )

        matrix = DMatrix(
            X,
            feature_names=list(X.columns)
        )

        progression = []

        for iteration in iterations:
            probability = float(
                booster.predict(
                    matrix,
                    iteration_range=(0, int(iteration))
                )[0]
            )

            progression.append({
                "iteration": int(iteration),
                "fraud_probability": round(
                    probability,
                    6
                )
            })

        return progression

    # ========================================================
    # LOCAL SHAP
    # ========================================================

    def get_local_shap(self, transaction, limit=20):
        try:
            import shap
        except ImportError:
            return []

        X = self.prepare_features(transaction)

        try:
            explainer = shap.TreeExplainer(self.model)
            values = explainer.shap_values(X)

            if isinstance(values, list):
                values = values[1]

            values = np.asarray(values)

            if values.ndim == 2:
                values = values[0]

            data = pd.DataFrame({
                "feature": X.columns,
                "shap_value": values
            })

            data["abs_shap"] = data["shap_value"].abs()

            data = data.sort_values(
                "abs_shap",
                ascending=False
            ).head(limit)

            return [
                {
                    "feature": str(row["feature"]),
                    "shap_value": round(
                        float(row["shap_value"]),
                        6
                    ),
                    "abs_shap": round(
                        float(row["abs_shap"]),
                        6
                    )
                }
                for _, row in data.iterrows()
            ]

        except Exception as e:
            print(f"⚠ Local SHAP failed: {e}")
            return []

    # ========================================================
    # RISK
    # ========================================================

    @staticmethod
    def classify_risk(risk):
        if risk >= 0.70:
            return {
                "risk_level": "CRITICAL",
                "decision": "REVERTED",
                "action": "BLOCK_TRANSACTION",
                "description": (
                    "High fraud risk detected. "
                    "Transaction should be blocked."
                )
            }

        if risk >= 0.40:
            return {
                "risk_level": "MODERATE",
                "decision": "MULTI-SIG REQUIRED",
                "action": "REQUIRE_ADDITIONAL_AUTHORIZATION",
                "description": (
                    "Moderate fraud risk detected. "
                    "Additional authorization is required."
                )
            }

        return {
            "risk_level": "LOW",
            "decision": "APPROVED",
            "action": "ALLOW_TRANSACTION",
            "description": (
                "Transaction falls within the acceptable risk range."
            )
        }

    # ========================================================
    # OFF-CHAIN RISK ENGINE
    # ========================================================

    @staticmethod
    def _clamp(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    @classmethod
    def calculate_offchain_risk(
        cls,
        verified_monthly_income,
        total_liabilities,
        requested_loan_amount,
        discrepancy_ratio
    ):
        """
        Converts Module 1 financial/document fields into a normalized
        0..1 off-chain risk score.

        This is a transparent heuristic layer, not a second ML model.
        The four source values are first converted into ratios so that
        different monetary scales do not get added directly.
        """
        income = max(float(verified_monthly_income or 0.0), 0.0)
        liabilities = max(float(total_liabilities or 0.0), 0.0)
        loan = max(float(requested_loan_amount or 0.0), 0.0)
        discrepancy = max(float(discrepancy_ratio or 0.0), 0.0)

        if income > 0:
            loan_to_income = loan / income
            liability_to_income = liabilities / income
        else:
            # Missing/zero verified income is treated as high financial risk.
            loan_to_income = 999.0 if loan > 0 else 0.0
            liability_to_income = 999.0 if liabilities > 0 else 0.0

        # Transparent normalization rules:
        # 0 risk at <=1x, maximum risk at >=5x loan/income.
        loan_risk = cls._clamp(
            (loan_to_income - 1.0) / 4.0
        )

        # 0 risk at <=1x, maximum risk at >=6x liabilities/income.
        liability_risk = cls._clamp(
            (liability_to_income - 1.0) / 5.0
        )

        # 0 risk at <=1.0 discrepancy, maximum at >=3.0.
        discrepancy_risk = cls._clamp(
            (discrepancy - 1.0) / 2.0
        )

        # Weighted off-chain score.
        offchain_risk = (
            0.40 * loan_risk
            + 0.30 * liability_risk
            + 0.30 * discrepancy_risk
        )

        return {
            "verified_monthly_income": round(income, 2),
            "total_liabilities": round(liabilities, 2),
            "requested_loan_amount": round(loan, 2),
            "discrepancy_ratio": round(discrepancy, 4),
            "loan_to_income_ratio": round(loan_to_income, 6),
            "liability_to_income_ratio": round(liability_to_income, 6),
            "loan_risk": round(loan_risk, 6),
            "liability_risk": round(liability_risk, 6),
            "discrepancy_risk": round(discrepancy_risk, 6),
            "off_chain_risk": round(
                cls._clamp(offchain_risk),
                6
            ),
            "weights": {
                "loan_risk": 0.40,
                "liability_risk": 0.30,
                "discrepancy_risk": 0.30
            }
        }

    # ========================================================
    # LIVE WALLET ASSESSMENT - HYBRID ON/OFF CHAIN
    # ========================================================

    def assess_wallet(
        self,
        borrower_address: str,
        verified_monthly_income=0.0,
        total_liabilities=0.0,
        requested_loan_amount=0.0,
        discrepancy_ratio=0.0,
        dr_penalty=0.0,
        include_shap=True,
        include_boosting=True
    ):
        lookup = self.find_wallet(borrower_address)

        if not lookup["found"]:
            return {
                "wallet_found": False,
                "borrower_address": borrower_address,
                "match_count": 0
            }

        transaction = lookup["transaction"]

        # -----------------------------
        # ON-CHAIN MODEL
        # -----------------------------
        prediction = self.predict(transaction)
        base_probability = prediction["fraud_probability"]
        on_chain_risk = self._clamp(base_probability)

        # -----------------------------
        # OFF-CHAIN MODEL
        # -----------------------------
        offchain = self.calculate_offchain_risk(
            verified_monthly_income=verified_monthly_income,
            total_liabilities=total_liabilities,
            requested_loan_amount=requested_loan_amount,
            discrepancy_ratio=discrepancy_ratio
        )

        off_chain_risk = offchain["off_chain_risk"]

        # -----------------------------
        # DR PENALTY
        # -----------------------------
        try:
            dr_penalty = float(dr_penalty or 0.0)
        except Exception:
            dr_penalty = 0.0

        if dr_penalty > 1:
            dr_penalty /= 100.0

        dr_penalty = self._clamp(dr_penalty)

        # -----------------------------
        # HYBRID RISK
        # -----------------------------
        # 70% learned on-chain signal
        # 30% transparent off-chain signal
        # plus Module 1's explicit DR penalty.
        on_chain_weight = 0.70
        off_chain_weight = 0.30

        weighted_on_chain = on_chain_weight * on_chain_risk
        weighted_off_chain = off_chain_weight * off_chain_risk

        final_risk = self._clamp(
            weighted_on_chain
            + weighted_off_chain
            + dr_penalty
        )

        classification = self.classify_risk(final_risk)

        result = {
            "wallet_found": True,
            "borrower_address": borrower_address,
            "matched_address": lookup["matched_address"],
            "match_count": lookup["match_count"],
            "dataset_index": lookup["dataset_index"],
            "wallet_transaction": transaction,

            "off_chain_input": offchain,

            "on_chain_risk": round(on_chain_risk, 6),
            "off_chain_risk": round(off_chain_risk, 6),
            "on_chain_weight": on_chain_weight,
            "off_chain_weight": off_chain_weight,
            "weighted_on_chain_risk": round(weighted_on_chain, 6),
            "weighted_off_chain_risk": round(weighted_off_chain, 6),

            "feature_pipeline": {
                "incoming_fields": len(transaction),
                "original_features": len(self.feature_columns),
                "features_after_imputation": len(self.feature_columns),
                "selected_features": len(self.selected_features)
            },

            "base_xgboost_probability": round(
                base_probability,
                6
            ),

            "dr_penalty": round(
                dr_penalty,
                6
            ),

            "final_risk": round(
                final_risk,
                6
            ),

            "final_risk_percentage": round(
                final_risk * 100,
                2
            ),

            "risk_formula": (
                "Final Risk = min(1, max(0, "
                "0.70*OnChainXGBoost + "
                "0.30*OffChainRisk + DRPenalty))"
            ),

            "model_prediction": prediction["prediction"],
            "threshold": self.threshold,
            "features_used": len(self.selected_features),

            **classification
        }

        if include_boosting:
            result["boosting_progression"] = (
                self.get_boosting_progression(transaction)
            )

        if include_shap:
            result["local_shap"] = (
                self.get_local_shap(transaction)
            )

        return result

    # ========================================================
    # STANDARD DIRECT TRANSACTION ASSESSMENT
    # ========================================================

    def assess(self, transaction, dr_penalty=0.0):
        prediction = self.predict(transaction)
        base_probability = prediction["fraud_probability"]

        try:
            dr_penalty = float(dr_penalty or 0.0)
        except Exception:
            dr_penalty = 0.0

        if dr_penalty > 1:
            dr_penalty /= 100.0

        dr_penalty = self._clamp(dr_penalty)
        final_risk = self._clamp(
            base_probability + dr_penalty
        )

        return {
            "base_xgboost_probability": round(base_probability, 6),
            "dr_penalty": round(dr_penalty, 6),
            "final_risk": round(final_risk, 6),
            "final_risk_percentage": round(final_risk * 100, 2),
            "model_prediction": prediction["prediction"],
            "threshold": prediction["threshold"],
            "features_used": prediction["features_used"],
            **self.classify_risk(final_risk)
        }

    # ========================================================
    # MODEL INFO
    # ========================================================

    def get_model_info(self):
        rounds = None

        try:
            rounds = (
                self.model
                .get_booster()
                .num_boosted_rounds()
            )
        except Exception:
            pass

        return {
            "model": "XGBoost",
            "boosting_rounds": rounds,
            "original_features": len(
                self.feature_columns
            ),
            "selected_features": len(
                self.selected_features
            ),
            "threshold": self.threshold,
            "dataset_loaded": self.dataset is not None,
            "dataset_rows": (
                len(self.dataset)
                if self.dataset is not None
                else 0
            ),
            "address_column": self.address_column,
            "model_loaded": self.model is not None,
            "imputer_loaded": self.imputer is not None,
            "shap_available": self.shap_features is not None,
            "hybrid_risk": {
                "on_chain_weight": 0.70,
                "off_chain_weight": 0.30,
                "dr_penalty": "added after weighted combination"
            }
        }

    # ========================================================
    # SHAP GLOBAL DATA
    # ========================================================

    def get_shap_data(self, limit=15):
        if (
            self.shap_features is None
            or self.shap_features.empty
        ):
            return []

        df = self.shap_features.copy()

        feature_column = next(
            (
                c for c in
                ["feature", "Feature", "FEATURE", "name"]
                if c in df.columns
            ),
            df.columns[0]
        )

        value_column = next(
            (
                c for c in
                [
                    "mean_abs_shap",
                    "mean_abs_shap_value",
                    "importance",
                    "Importance",
                    "mean_abs"
                ]
                if c in df.columns
            ),
            None
        )

        if value_column is None:
            numerical = df.select_dtypes(
                include=[np.number]
            ).columns

            if len(numerical) == 0:
                return []

            value_column = numerical[0]

        df[value_column] = pd.to_numeric(
            df[value_column],
            errors="coerce"
        )

        df = df.dropna(
            subset=[value_column]
        )

        df = (
            df.sort_values(
                value_column,
                ascending=False
            )
            .head(limit)
        )

        return [
            {
                "feature": str(row[feature_column]),
                "importance": round(
                    float(row[value_column]),
                    6
                )
            }
            for _, row in df.iterrows()
        ]


_model_service = None


def get_model_service():
    global _model_service

    if _model_service is None:
        _model_service = DeFiLensModel()

    return _model_service
