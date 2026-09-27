import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
print("GEMINI KEY LOADED:", bool(os.getenv("GEMINI_API_KEY")))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Gemini GenAI
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="DeFiLens | Live Fraud Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / URLS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "transaction_dataset.csv"
MODULE1_PAYLOAD_PATH = BASE_DIR / "latest_module1_payload.json"
MODULE2_PAYLOAD_PATH = BASE_DIR / "latest_module2_payload.json"

API_URL = "http://localhost:8001"
MODULE3_URL = "http://localhost:5000/api/blockchain/process-transaction"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(80,95,190,.10), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(0,180,170,.06), transparent 25%),
        #080b12;
    color: #e8edf7;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #0b0f18;
    border-right: 1px solid #1c2433;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #293448;
    background: #111827;
    color: #e8edf7;
    font-weight: 700;
    min-height: 42px;
}

button[kind="primary"] {
    background: linear-gradient(135deg,#5865f2,#667eea) !important;
    border: none !important;
}

[data-testid="stMetric"] {
    background: #0e131e;
    border: 1px solid #1d2737;
    border-radius: 12px;
    padding: 14px 17px;
}

.card {
    background: linear-gradient(145deg,rgba(17,24,39,.96),rgba(11,15,24,.98));
    border: 1px solid #202b3d;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.card-title {
    font-size: 11px;
    color: #8d99ae;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.card-value {
    font-size: 25px;
    font-weight: 800;
    color: #f4f7fb;
    margin-top: 6px;
}

.card-subtitle {
    font-size: 12px;
    color: #778399;
    margin-top: 4px;
}

.layer-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 12px 0 8px;
}

.layer-number {
    width: 35px;
    height: 35px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #151d2e;
    border: 1px solid #33415c;
    color: #8f9bff;
    font-weight: 800;
}

.layer-name {
    font-size: 19px;
    font-weight: 800;
}

.layer-sub {
    font-size: 11px;
    color: #738097;
}

.pipeline {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 16px 0 24px;
}

.pipeline-node {
    flex: 1;
    text-align: center;
    padding: 14px 7px;
    border-radius: 11px;
    background: #101621;
    border: 1px solid #263246;
}

.pipeline-node.active {
    border-color: #5966e8;
}

.pipeline-title {
    font-size: 11px;
    font-weight: 800;
}

.pipeline-sub {
    font-size: 9px;
    color: #748096;
    margin-top: 2px;
}

.pipeline-arrow {
    color: #536078;
    font-size: 18px;
}

.risk-box {
    border-radius: 15px;
    padding: 23px;
    background: #0d131e;
    border: 1px solid #263247;
    text-align: center;
}

.risk-number {
    font-size: 51px;
    line-height: 1;
    font-weight: 800;
    margin: 12px 0;
}

.small-label {
    color: #778399;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-weight: 800;
}

hr {
    border-color: #1c2534 !important;
}
</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def safe(value):
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def load_json(path, default=None):
    if default is None:
        default = {}

    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def risk_info(risk):
    if risk >= 0.70:
        return (
            "CRITICAL",
            "REVERTED",
            "BLOCKED",
            "#ff5f6d"
        )

    if risk >= 0.40:
        return (
            "MODERATE",
            "MULTI-SIG REQUIRED",
            "REVIEW",
            "#f2b84b"
        )

    return (
        "LOW",
        "APPROVED",
        "FORWARD",
        "#45d483"
    )


def make_risk_meter(risk):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-.2, 1.15)
    ax.axis("off")

    theta = np.linspace(np.pi, 0, 300)

    ax.plot(
        np.cos(theta),
        np.sin(theta),
        linewidth=24,
        alpha=.12
    )

    end = np.pi * (1 - risk)

    fill = np.linspace(
        np.pi,
        end,
        200
    )

    ax.plot(
        np.cos(fill),
        np.sin(fill),
        linewidth=24,
        alpha=.95
    )

    needle = np.pi * (1 - risk)

    ax.plot(
        [0, .78 * np.cos(needle)],
        [0, .78 * np.sin(needle)],
        linewidth=3
    )

    ax.scatter(
        [0],
        [0],
        s=160,
        zorder=5
    )

    ax.text(
        0,
        .30,
        f"{risk * 100:.2f}%",
        ha="center",
        va="center",
        fontsize=29,
        fontweight="bold"
    )

    ax.text(
        0,
        .05,
        "FINAL RISK",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()

    return fig


def api_health():
    try:
        response = requests.get(
            f"{API_URL}/api/v1/health",
            timeout=30
        )
        return response.ok
    except Exception:
        return False


def run_live_assessment(
    payload_data
):
    payload = {
        "borrower_wallet_address": payload_data.get("borrower_wallet_address", ""),
        "verified_monthly_income": payload_data.get("verified_monthly_income", 0.0),
        "total_liabilities": payload_data.get("total_liabilities", 0.0),
        "requested_loan_amount": payload_data.get("requested_loan_amount", 0.0),
        "discrepancy_ratio": payload_data.get("discrepancy_ratio", 0.0),
        "dr_risk_penalty": payload_data.get("dr_risk_penalty", 0.0),
        "include_shap": True,
        "include_boosting": True,
        "forward_to_module3": True,
        "forward_to_gateway": False
    }

    try:
        response = requests.post(
            f"{API_URL}/api/v1/assess-risk",
            json=safe(payload),
            timeout=45
        )

        if response.ok:
            return {
                "success": True,
                "data": response.json()
            }

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        return {
            "success": False,
            "error": detail,
            "status": response.status_code
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status": None
        }


def forward_to_module3(payload):
    try:
        response = requests.post(
            MODULE3_URL,
            json=safe(payload),
            timeout=15
        )

        try:
            data = response.json()
        except Exception:
            data = {"raw_response": response.text}

        return {
            "success": response.ok,
            "status": response.status_code,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "status": None,
            "data": {"error": str(e)}
        }


def make_global_shap_chart(rows):
    if not rows:
        return None

    data = pd.DataFrame(rows)

    if data.empty:
        return None

    data = data.head(15).sort_values(
        "importance"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(
        data["feature"],
        data["importance"]
    )

    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP Feature Importance")

    plt.tight_layout()

    return fig


def make_local_shap_chart(rows):
    if not rows:
        return None

    data = pd.DataFrame(rows)

    if data.empty:
        return None

    data = data.head(15).sort_values(
        "shap_value"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(
        data["feature"],
        data["shap_value"]
    )

    ax.axvline(
        0,
        linewidth=1,
        alpha=.6
    )

    ax.set_xlabel("SHAP contribution")
    ax.set_title(
        "Current Borrower — Local SHAP Explanation"
    )

    plt.tight_layout()

    return fig


def simple_feature_name(feature):
    """Convert technical Ethereum dataset feature names into simple language."""
    name = str(feature).strip()
    key = name.lower().replace("_", " ")

    descriptions = {
        "sent tnx": "number of transactions sent",
        "received tnx": "number of transactions received",
        "total transactions including tnx to create contract": "total number of transactions",
        "total transactions": "total number of transactions",
        "total ether sent": "total amount of ETH sent",
        "total ether received": "total amount of ETH received",
        "total ether balance": "ETH balance",
        "number of created contracts": "number of contracts created",
        "unique received from addresses": "number of different addresses sending money to this wallet",
        "unique sent to addresses": "number of different addresses this wallet sends money to",
        "avg min between sent tnx": "average time between sent transactions",
        "avg min between received tnx": "average time between received transactions",
        "time diff between first and last (mins)": "activity period of the wallet",
        "erc20 total tnx": "number of ERC-20 token transactions",
        "erc20 total transactions": "number of ERC-20 token transactions",
        "erc20 uniq sent addr": "number of different addresses receiving tokens from this wallet",
        "erc20 uniq rec addr": "number of different addresses sending tokens to this wallet",
        "erc20 total ether received": "total ERC-20 tokens received",
        "erc20 total ether sent": "total ERC-20 tokens sent",
        "erc20 total ether balance": "ERC-20 token balance",
        "min value received": "minimum amount received in one transaction",
        "max value received": "maximum amount received in one transaction",
        "avg value received": "average amount received per transaction",
        "min value sent": "minimum amount sent in one transaction",
        "max value sent": "maximum amount sent in one transaction",
        "avg value sent": "average amount sent per transaction",
    }

    if key in descriptions:
        return descriptions[key]

    # General fallbacks for common Ethereum feature naming patterns.
    if "unique" in key and "received" in key:
        return "number of different addresses sending money to this wallet"
    if "unique" in key and "sent" in key:
        return "number of different addresses this wallet sends money to"
    if "avg" in key and "sent" in key:
        return "average amount or time for sent transactions"
    if "avg" in key and "received" in key:
        return "average amount or time for received transactions"
    if "max" in key and "sent" in key:
        return "largest amount sent"
    if "max" in key and "received" in key:
        return "largest amount received"
    if "min" in key and "sent" in key:
        return "smallest amount sent"
    if "min" in key and "received" in key:
        return "smallest amount received"
    if "erc20" in key:
        return "ERC-20 token activity"
    if "gas" in key:
        return "gas usage"
    if "transaction" in key or "tnx" in key:
        return "transaction activity"

    return name.replace("_", " ").strip()


def simple_shap_reason(feature, shap_value, fraud_prediction):
    """Create a simple-language explanation without exposing SHAP jargon."""
    description = simple_feature_name(feature)

    if shap_value > 0:
        if fraud_prediction:
            return (
                f"**{description}** made the wallet look more suspicious to "
                f"the model."
            )
        return (
            f"**{description}** increased the fraud score, but other safer "
            f"signals kept the overall prediction below the fraud threshold."
        )

    if fraud_prediction:
        return (
            f"**{description}** was a safer signal and reduced the fraud "
            f"score, but it was not strong enough to outweigh the suspicious signals."
        )

    return (
        f"**{description}** made the wallet look more normal to the model "
        f"and helped keep the prediction on the legitimate side."
    )


def render_local_shap_explanation(rows, fraud_probability, threshold=0.5):
    """Show a simple-language explanation of the current XGBoost prediction."""
    if not rows:
        return

    data = pd.DataFrame(rows).copy()

    if (
        data.empty
        or "feature" not in data.columns
        or "shap_value" not in data.columns
    ):
        st.info("A simple explanation is not available for this prediction.")
        return

    data["shap_value"] = pd.to_numeric(
        data["shap_value"],
        errors="coerce"
    )

    data = data.dropna(subset=["shap_value"])

    if data.empty:
        st.info("A simple explanation is not available for this prediction.")
        return

    data["feature"] = data["feature"].astype(str)
    data["abs_shap"] = data["shap_value"].abs()
    data = data.sort_values("abs_shap", ascending=False)

    positive = data[data["shap_value"] > 0].head(3)
    negative = (
        data[data["shap_value"] < 0]
        .sort_values("shap_value")
        .head(3)
    )

    is_fraud = fraud_probability >= threshold

    st.markdown("### 🧠 Why did the model make this prediction?")

    if is_fraud:
        st.error(
            f"**The model considers this transaction suspicious.** "
            f"The fraud probability is **{fraud_probability * 100:.2f}%**, "
            f"which is above the **{threshold * 100:.0f}%** fraud threshold."
        )
    else:
        st.success(
            f"**The model considers this transaction likely legitimate.** "
            f"The fraud probability is **{fraud_probability * 100:.2f}%**, "
            f"which is below the **{threshold * 100:.0f}%** fraud threshold."
        )

    st.markdown("#### 🔴 What made it look suspicious?")

    if positive.empty:
        st.info("No strong suspicious signals were found.")
    else:
        for _, row in positive.iterrows():
            st.markdown(
                "• "
                + simple_shap_reason(
                    row["feature"],
                    row["shap_value"],
                    is_fraud
                )
            )

    st.markdown("#### 🟢 What made it look safer?")

    if negative.empty:
        st.info("No strong safer signals were found.")
    else:
        for _, row in negative.iterrows():
            st.markdown(
                "• "
                + simple_shap_reason(
                    row["feature"],
                    row["shap_value"],
                    is_fraud
                )
            )

    top_positive = positive.iloc[0] if not positive.empty else None
    top_negative = negative.iloc[0] if not negative.empty else None

    st.markdown("#### 📌 Simple explanation")

    if is_fraud:
        if top_positive is not None:
            main_reason = (
                f"The main reason is that **"
                f"{simple_feature_name(top_positive['feature'])}** "
                f"was a strong suspicious signal."
            )
        else:
            main_reason = (
                "The model found several transaction patterns associated "
                "with suspicious activity."
            )

        if top_negative is not None:
            counter_reason = (
                f"Some safer behaviour, such as **"
                f"{simple_feature_name(top_negative['feature'])}**, "
                f"reduced the score slightly."
            )
        else:
            counter_reason = ""

        st.info(
            f"{main_reason} {counter_reason} "
            f"After combining all these signals, the model estimated a "
            f"**{fraud_probability * 100:.2f}%** chance of fraud."
        )

    else:
        if top_negative is not None:
            main_reason = (
                f"The main reason is that **"
                f"{simple_feature_name(top_negative['feature'])}** "
                f"was a strong safe signal."
            )
        else:
            main_reason = (
                "The model found more normal transaction behaviour than "
                "suspicious behaviour."
            )

        if top_positive is not None:
            counter_reason = (
                f"However, **{simple_feature_name(top_positive['feature'])}** "
                f"added some suspicion."
            )
        else:
            counter_reason = ""

        st.info(
            f"{main_reason} {counter_reason} "
            f"After combining all the signals, the model estimated only a "
            f"**{fraud_probability * 100:.2f}%** chance of fraud, so the "
            f"transaction remained on the legitimate side."
        )

    with st.expander("What does this explanation mean?"):
        st.markdown(
            """
**In simple terms:**

The model looks at many transaction patterns at the same time.

- 🔴 **Suspicious signal:** a feature that increases the fraud score.
- 🟢 **Safe signal:** a feature that decreases the fraud score.
- The model combines all these signals to produce the final fraud probability.
- The explanation above shows the **strongest reasons for this particular wallet**.

So the system does not simply say *"Fraud"* or *"Legitimate"* — it also shows **what transaction behaviour influenced that decision**.
"""
        )


# ============================================================
# GEMINI GENAI EXPLANATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash"


FEATURE_NAME_MAP = {
    "sent tnx": "transactions sent",
    "received tnx": "transactions received",
    "total transactions including tnx to create contract": "total transactions",
    "total transactions": "total transactions",
    "total ether sent": "ETH sent",
    "total ether received": "ETH received",
    "total ether balance": "ETH balance",
    "number of created contracts": "contracts created",
    "unique received from addresses": "different addresses sending money to this wallet",
    "unique sent to addresses": "different addresses this wallet sends money to",
    "avg min between sent tnx": "average time between sent transactions",
    "avg min between received tnx": "average time between received transactions",
    "time diff between first and last (mins)": "wallet activity period",
    "erc20 total tnx": "ERC-20 token transactions",
    "erc20 total transactions": "ERC-20 token transactions",
    "erc20 uniq sent addr": "different addresses receiving tokens",
    "erc20 uniq rec addr": "different addresses sending tokens",
    "erc20 total ether received": "ERC-20 tokens received",
    "erc20 total ether sent": "ERC-20 tokens sent",
    "erc20 total ether balance": "ERC-20 token balance",
    "min value received": "smallest amount received",
    "max value received": "largest amount received",
    "avg value received": "average amount received",
    "min value sent": "smallest amount sent",
    "max value sent": "largest amount sent",
    "avg value sent": "average amount sent",
}


def simple_feature_name(feature):
    """Turn a dataset column into natural user-facing language."""
    key = str(feature).strip().lower().replace("_", " ")

    if key in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[key]

    if "unique" in key and "received" in key:
        return "different addresses sending money to this wallet"

    if "unique" in key and "sent" in key:
        return "different addresses this wallet sends money to"

    if "max" in key and "sent" in key:
        return "largest amount sent"

    if "max" in key and "received" in key:
        return "largest amount received"

    if "min" in key and "sent" in key:
        return "smallest amount sent"

    if "min" in key and "received" in key:
        return "smallest amount received"

    if "avg" in key and "sent" in key:
        return "average sent transaction behaviour"

    if "avg" in key and "received" in key:
        return "average received transaction behaviour"

    if "erc20" in key:
        return "ERC-20 token activity"

    if "gas" in key:
        return "gas usage"

    if "transaction" in key or "tnx" in key:
        return "transaction activity"

    return str(feature).replace("_", " ").strip()


def _feature_value(wallet_transaction, feature):
    """Find the observed wallet value for a SHAP feature when available."""
    if not isinstance(wallet_transaction, dict):
        return None

    feature_str = str(feature)
    if feature_str in wallet_transaction:
        return wallet_transaction[feature_str]

    feature_lower = feature_str.lower()
    for key, value in wallet_transaction.items():
        if str(key).lower() == feature_lower:
            return value

    return None


def _format_observed_value(feature, value):
    """Keep values readable without making the explanation technical."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return None

    key = str(feature).lower()

    if "mins" in key or "min between" in key:
        return f"{number:,.1f} minutes"

    if "ether" in key or "value" in key or "balance" in key:
        return f"{number:,.4f}"

    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"

    return f"{number:,.2f}"


def get_gemini_signals(
    local_shap,
    fraud_prediction,
    wallet_transaction=None,
    limit=4
):
    """
    SHAP is used only to select the strongest signals.
    Gemini receives simple descriptions and, when available,
    the observed wallet value.
    """
    signals = []

    if not isinstance(local_shap, list):
        return signals

    for row in local_shap:

        if not isinstance(row, dict):
            continue

        feature = row.get("feature")

        if not feature:
            continue

        try:
            shap_value = float(
                row.get("shap_value", 0.0)
            )
        except (TypeError, ValueError):
            continue

        # Positive SHAP pushes toward fraud.
        # Negative SHAP pushes toward benign.
        if fraud_prediction and shap_value > 0:
            direction = "suspicious"
        elif not fraud_prediction and shap_value < 0:
            direction = "safe"
        else:
            continue

        observed = _feature_value(
            wallet_transaction,
            feature
        )

        signals.append(
            {
                "signal": simple_feature_name(feature),
                "direction": direction,
                "observed_value": _format_observed_value(
                    feature,
                    observed
                ),
                "strength": abs(shap_value),
            }
        )

    signals.sort(
        key=lambda item: item["strength"],
        reverse=True
    )

    return signals[:limit]


def fallback_genai_explanation(
    local_shap,
    fraud_probability,
    fraud_prediction,
    wallet_transaction=None
):
    """
    Natural-language fallback used only if Gemini cannot be reached.
    It is deliberately written differently from the old rigid template.
    """
    signals = get_gemini_signals(
        local_shap,
        fraud_prediction,
        wallet_transaction,
        limit=4
    )

    if fraud_prediction:

        title = "Why is this transaction suspicious?"

        points = []

        for item in signals[:3]:

            value_text = (
                f" ({item['observed_value']})"
                if item.get("observed_value")
                else ""
            )

            points.append(
                f"The wallet's {item['signal']}{value_text} "
                "looks unusual and pushed the risk higher."
            )

        if not points:
            points = [
                "The wallet shows behaviour that is less typical of "
                "the legitimate transactions seen by the model.",
                "Several transaction patterns together pushed the "
                "fraud score higher.",
                f"The model estimated a {fraud_probability * 100:.2f}% "
                "chance of fraud."
            ]

    else:

        title = "Why is this transaction considered safe?"

        points = []

        for item in signals[:3]:

            value_text = (
                f" ({item['observed_value']})"
                if item.get("observed_value")
                else ""
            )

            points.append(
                f"The wallet's {item['signal']}{value_text} "
                "looks more like normal activity and lowered the risk."
            )

        if not points:
            points = [
                "The wallet shows transaction behaviour that is closer "
                "to the legitimate activity learned by the model.",
                "The model did not find strong signals pointing toward fraud.",
                f"The model estimated only a {fraud_probability * 100:.2f}% "
                "chance of fraud."
            ]

    return {
        "title": title,
        "points": points[:3],
        "source": "Fallback explanation"
    }


def generate_genai_explanation(local_shap, fraud_probability, fraud_prediction, wallet_transaction=None):
    """XGBoost decides; SHAP selects signals; Gemini explains them."""
    fallback = fallback_genai_explanation(
        local_shap, fraud_probability, fraud_prediction, wallet_transaction
    )

    if genai is None or types is None:
        print("GEMINI ERROR: google-genai SDK unavailable.")
        return fallback

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI ERROR: GEMINI_API_KEY is not set.")
        return fallback

    signals = get_gemini_signals(
        local_shap, fraud_prediction, wallet_transaction, limit=4
    )
    if not signals:
        print("GEMINI WARNING: No suitable SHAP signals.")
        return fallback

    safe_signals = []
    for item in signals:
        s = {"behaviour": item["signal"], "effect": item["direction"]}
        if item.get("observed_value") is not None:
            s["observed_value"] = item["observed_value"]
        safe_signals.append(s)

    if fraud_prediction:
        title = "Why is this transaction suspicious?"
        decision = "SUSPICIOUS / FRAUD"
        focus = """The transaction has already been classified as SUSPICIOUS.
ONLY explain why the supplied behaviour looks suspicious or unusual.
Every point MUST support the suspicious classification.
Do NOT describe anything as safe, normal, legitimate, or low-risk."""
    else:
        title = "Why is this transaction considered safe?"
        decision = "BENIGN / LIKELY LEGITIMATE"
        focus = """The transaction has already been classified as LIKELY LEGITIMATE.
ONLY explain why the supplied behaviour looks normal, consistent, or safer.
Every point MUST support the legitimate classification.
Do NOT describe anything as suspicious, fraudulent, unusual, or high-risk."""

    prompt = f"""
You are the natural-language explanation component of a DeFi fraud detection dashboard.

The fraud detector has ALREADY made the decision.

Decision: {decision}
Fraud probability: {fraud_probability * 100:.2f}%

Strongest supplied signals:
{json.dumps(safe_signals, indent=2)}

{focus}

Rules:
- Generate exactly 3 short points.
- Each point must be one clear sentence.
- Use simple language for an ordinary DeFi user.
- Explain the supplied behaviour and why it matters.
- Do not mention SHAP, XGBoost, machine learning, features, weights,
  algorithms, or model internals.
- Do not invent facts.
- Do not change the supplied decision.
- Never say the transaction is definitely fraudulent.
- Never say a legitimate transaction is guaranteed safe.
- Do not include the fraud probability in the points.
- Return ONLY the three points, one per line.
- Do not number the points.
"""

    try:
        print("========== GEMINI EXPLANATION ==========")
        print("Decision:", decision)
        print("Signals:", safe_signals)

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=180
            )
        )

        text = (getattr(response, "text", None) or "").strip()
        print("Gemini response:", repr(text))

        points = []

        for line in text.splitlines():
            line = re.sub(r"^[-•*]\s*", "", line.strip())
            line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

            if len(line) >= 10:
                points.append(line)

        # If Gemini returns fewer than 3 points,
        # ask it once more to format the answer correctly.
        if len(points) < 3:

            retry_prompt = f"""
Rewrite this explanation into exactly 3 short points.

Decision: {decision}

Original explanation:
{text}

Rules:
- Exactly 3 points
- One sentence per point
- Use very simple language
- Support the existing decision
- Do not change the decision
- Do not mention SHAP, XGBoost, machine learning,
  features, weights, or algorithms
- Do not invent information
- Return only the 3 points
"""

            retry_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=retry_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=250
                )
            )

            retry_text = (
                getattr(retry_response, "text", None) or ""
            ).strip()

            print(
                "Gemini retry response:",
                repr(retry_text)
            )

            points = []

            for line in retry_text.splitlines():
                line = re.sub(
                    r"^[-•*]\s*",
                    "",
                    line.strip()
                )
                line = re.sub(
                    r"^\d+[\.\)]\s*",
                    "",
                    line
                ).strip()

                if len(line) >= 10:
                    points.append(line)

        if len(points) < 3:
            raise RuntimeError(
                f"Gemini returned only {len(points)} usable points."
            )

        return {
            "title": title,
            "points": points[:3],
            "source": "Gemini GenAI"
        }

    except Exception as error:
        print("========== GEMINI ACTUAL ERROR ==========")
        print(type(error).__name__)
        print(repr(error))
        print("==========================================")
        return fallback

def render_genai_explanation(explanation):
    """
    Render the Gemini explanation on the Overview page.
    """
    if not explanation:
        return

    title = explanation.get(
        "title",
        "AI Explanation"
    )

    points = explanation.get(
        "points",
        []
    )

    source = explanation.get(
        "source",
        "AI"
    )

    if title.startswith(
        "Why is this transaction suspicious?"
    ):

        st.markdown(
            "### 🔴 Why is this transaction suspicious?"
        )

    else:

        st.markdown(
            "### 🟢 Why is this transaction considered safe?"
        )

    for index, point in enumerate(
        points,
        start=1
    ):

        st.markdown(
            f"""
<div style="
    padding:15px 17px;
    margin:9px 0;
    border-radius:11px;
    border:1px solid #202b3d;
    background:#0e131e;
">
    <div style="display:flex;gap:13px;align-items:flex-start;">
        <div style="
            min-width:28px;
            height:28px;
            border-radius:50%;
            background:#182237;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight:800;
            font-size:12px;
        ">
            {index}
        </div>
        <div style="
            line-height:1.55;
            padding-top:2px;
        ">
            {point}
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True
        )

    # Do not make the user-facing explanation sound like a technical
    # SHAP report. Keep the source information subtle.
    if source == "Gemini GenAI":
        st.caption(
            "Generated by Gemini from the strongest transaction signals. "
            "The fraud decision itself comes from the detection model."
        )
    else:
        st.caption(
            "Gemini was unavailable, so a local explanation was shown instead. "
            "The fraud decision itself comes from the detection model."
        )


def normalize_boosting_progression(rows):
    cleaned=[]
    for row in rows or []:
        try:
            iteration=int(row.get("iteration")); probability=float(row.get("fraud_probability"))
        except (TypeError,ValueError,AttributeError): continue
        if np.isfinite(probability): cleaned.append({"iteration":iteration,"fraud_probability":max(0.0,min(1.0,probability))})
    return sorted(cleaned,key=lambda x:x["iteration"])


def get_boosting_checkpoints(rows):
    rows=normalize_boosting_progression(rows)
    if not rows: return []
    by_iteration={r["iteration"]:r for r in rows}; final_iteration=rows[-1]["iteration"]
    selected=[]
    for iteration in [100,200,300,400,500,final_iteration]:
        if iteration in by_iteration: row=by_iteration[iteration]
        elif iteration==final_iteration: row=rows[-1]
        else:
            candidates=[r for r in rows if r["iteration"]<=iteration]
            if not candidates: continue
            row=candidates[-1]
        if not selected or row["iteration"]!=selected[-1]["iteration"]: selected.append(row)
    result=[]
    for i,row in enumerate(selected):
        prev=selected[i-1]["fraud_probability"] if i else None
        result.append({"iteration":row["iteration"],"fraud_probability":row["fraud_probability"],"change":None if prev is None else row["fraud_probability"]-prev})
    return result


def checkpoint_contribution_text(checkpoint, previous=None):
    iteration=checkpoint["iteration"]; probability=checkpoint["fraud_probability"]
    if previous is None:
        return f"After the first {iteration} boosting trees, the XGBoost ensemble produces a cumulative fraud probability of {probability*100:.2f}%. This is the first displayed checkpoint."
    change=checkpoint["change"]; direction="increased" if change>=0 else "decreased"
    return (f"Trees {previous['iteration']+1}–{iteration} {direction} the ensemble probability by {abs(change)*100:.2f} percentage points, "
            f"moving the cumulative fraud probability from {previous['fraud_probability']*100:.2f}% to {probability*100:.2f}%. "
            "These trees refine the existing ensemble by learning from errors/residuals left by the previous trees.")


def make_boosting_chart(rows, threshold):
    rows=normalize_boosting_progression(rows)
    if not rows: return None
    data=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.plot(data["iteration"],data["fraud_probability"]*100,linewidth=2.5)
    checkpoints=get_boosting_checkpoints(rows)
    if checkpoints:
        ax.scatter([r["iteration"] for r in checkpoints],[r["fraud_probability"]*100 for r in checkpoints],s=55,zorder=4)
        for r in checkpoints:
            ax.annotate(f"{r['iteration']}\n{r['fraud_probability']*100:.2f}%",(r["iteration"],r["fraud_probability"]*100),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8)
    ax.axhline(threshold*100,linestyle="--",linewidth=1.5,alpha=.6)
    ax.set_xlabel("Boosting Tree"); ax.set_ylabel("Cumulative Fraud Probability (%)"); ax.set_title("XGBoost Probability Through Boosting"); ax.grid(alpha=.15,linestyle="--")
    plt.tight_layout(); return fig


# ============================================================
# LOAD MODULE 1 PAYLOAD
# ============================================================

module1_payload = load_json(
    MODULE1_PAYLOAD_PATH,
    {}
)

# Module 2 writes this file after processing. It lets the dashboard
# display the same result that was automatically sent to Module 3.
latest_module2_payload = load_json(
    MODULE2_PAYLOAD_PATH,
    {}
)

# Support several sensible naming conventions from Module 1.
module1_address = (
    module1_payload.get("borrower_address")
    or module1_payload.get("borrowerAddress")
    or module1_payload.get("wallet_address")
    or module1_payload.get("walletAddress")
    or module1_payload.get("borrower_wallet_address")
    or ""
)

module1_penalty = (
    module1_payload.get("dr_risk_penalty")
    if module1_payload.get("dr_risk_penalty") is not None
    else (
        module1_payload.get("dr_penalty")
        or module1_payload.get("risk_penalty")
        or 0.0
    )
)

try:
    module1_penalty = float(module1_penalty)
except Exception:
    module1_penalty = 0.0

if module1_penalty > 1:
    module1_penalty /= 100.0

module1_penalty = max(
    0.0,
    min(1.0, module1_penalty)
)


# ============================================================
# SESSION STATE
# ============================================================

if "live_assessment" not in st.session_state:
    st.session_state.live_assessment = None

if "live_address" not in st.session_state:
    st.session_state.live_address = module1_address


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div style="display:flex;align-items:center;gap:11px;">
    <div style="width:40px;height:40px;border-radius:10px;
                background:linear-gradient(135deg,#5865f2,#7b83ff);
                display:flex;align-items:center;justify-content:center;
                font-size:21px;font-weight:800;">
        ◈
    </div>
    <div>
        <div style="font-size:23px;font-weight:800;">DeFiLens</div>
        <div style="font-size:11px;color:#78859b;">
            Live Transaction Intelligence
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "◈ Live Risk Pipeline",
            "① Module 1 · Payload",
            "② Wallet → XGBoost",
            "③ SHAP · Explainability",
            "④ Module 3 · Handoff",
            "Model Performance"
        ]
    )

    st.markdown("---")

    online = api_health()

    st.markdown(
        f"""
<div class="status-pill">
    <span class="status-dot"
          style="background:{'#45d483' if online else '#ff5f6d'};
                 width:7px;height:7px;border-radius:50%;
                 display:inline-block;"></span>
    {'API ONLINE' if online else 'API OFFLINE'}
</div>
""",
        unsafe_allow_html=True
    )

    if module1_address:
        st.markdown("---")
        st.caption("Module 1 borrower")
        st.code(module1_address)


# ============================================================
# HEADER
# ============================================================

left, right = st.columns(
    [7, 2],
    vertical_alignment="center"
)

with left:
    st.markdown(
        """
<div>
    <div style="font-size:12px;color:#78859b;text-transform:uppercase;
                letter-spacing:.12em;font-weight:800;">
        DeFi Security Intelligence
    </div>
    <div style="font-size:31px;font-weight:800;margin-top:4px;">
        Live Transaction Risk Center
    </div>
    <div style="color:#78859b;margin-top:5px;font-size:13px;">
        Module 1 → Wallet Lookup → XGBoost → GenAI Explanation → Risk Engine → Module 3
    </div>
</div>
""",
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        """
<div style="text-align:right;">
    <div class="status-pill">
        <span class="status-dot"></span>
        LIVE PIPELINE
    </div>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown("---")


# ============================================================
# LIVE INPUT — MODULE 1
# ============================================================

st.markdown(
    '<div class="section-title">Live Module 1 Input</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'The borrower address is received from Module 1 and used to locate '
    'the corresponding wallet record in transaction_dataset.csv.'
    '</div>',
    unsafe_allow_html=True
)

input_left, input_right = st.columns(
    [3, 1]
)

with input_left:

    borrower_address = module1_address.strip()

    st.text_input(
        "Borrower wallet address received from Module 1",
        value=borrower_address,
        disabled=True,
        help="This value is read from the latest payload received from Module 1."
    )

with input_right:

    st.metric(
        "Module 1 DR Penalty",
        f"+{module1_penalty * 100:.2f}%"
    )

st.markdown("### Off-chain financial inputs")
o1, o2, o3, o4 = st.columns(4)
with o1:
    st.metric("Monthly Income", f"₹{float(module1_payload.get('verified_monthly_income', 0) or 0):,.0f}")
with o2:
    st.metric("Liabilities", f"₹{float(module1_payload.get('total_liabilities', 0) or 0):,.0f}")
with o3:
    st.metric("Requested Loan", f"₹{float(module1_payload.get('requested_loan_amount', 0) or 0):,.0f}")
with o4:
    st.metric("Discrepancy Ratio", f"{float(module1_payload.get('discrepancy_ratio', 0) or 0):.2f}x")

refresh_input = st.button(
    "↻ REFRESH MODULE 1 INPUT",
    use_container_width=True
)

if refresh_input:
    st.rerun()

run = st.button(
    "PROCESS LATEST MODULE 1 INPUT",
    type="primary",
    use_container_width=True
)

if run:

    if not borrower_address.strip():
        st.error(
            "Module 1 did not provide a borrower address."
        )
    elif not online:
        st.error(
            "FastAPI Module 2 is offline. Start uvicorn first."
        )
    else:

        with st.spinner(
            "Module 2: locating wallet, preparing features, "
            "running XGBoost and preparing the AI explanation..."
        ):

            process_payload = {
                **module1_payload,
                "borrower_wallet_address": borrower_address.strip(),
                "dr_risk_penalty": module1_penalty
            }

            result = run_live_assessment(
                process_payload
            )

        if result["success"]:

            assessment_result = result["data"]["assessment"]

            # XGBoost remains the actual decision-maker.
            fraud_probability = float(
                assessment_result.get(
                    "base_xgboost_probability",
                    0.0
                ) or 0.0
            )

            model_threshold = float(
                assessment_result.get(
                    "threshold",
                    0.5
                ) or 0.5
            )

            fraud_prediction = (
                fraud_probability >= model_threshold
            )

            # SHAP is used internally to identify the strongest
            # signals. Gemini converts those signals into simple
            # natural-language points.
            local_shap = assessment_result.get(
                "local_shap",
                []
            )

            with st.spinner(
                "Generating simple AI explanation..."
            ):
                assessment_result["genai_explanation"] = (
                    generate_genai_explanation(
                        local_shap=local_shap,
                        fraud_probability=fraud_probability,
                        fraud_prediction=fraud_prediction,
                        wallet_transaction=assessment_result.get(
                            "wallet_transaction",
                            {}
                        ),
                    )
                )

            st.session_state.live_assessment = (
                assessment_result
            )

            st.session_state.live_address = (
                borrower_address.strip()
            )

            st.success(
                "Live borrower assessment completed."
            )

        else:

            st.session_state.live_assessment = None

            st.error(
                f"Module 2 could not process this borrower: "
                f"{result['error']}"
            )


assessment = st.session_state.live_assessment


# ============================================================
# FULL LIVE PIPELINE
# ============================================================

if page == "◈ Live Risk Pipeline":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Live Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="pipeline">
    <div class="pipeline-node active">
        <div>◉</div>
        <div class="pipeline-title">MODULE 1</div>
        <div class="pipeline-sub">Borrower Input</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node active">
        <div>⌕</div>
        <div class="pipeline-title">WALLET LOOKUP</div>
        <div class="pipeline-sub">Dataset Match</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node active">
        <div>◆</div>
        <div class="pipeline-title">XGBOOST</div>
        <div class="pipeline-sub">Fraud Detection</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node active">
        <div>✦</div>
        <div class="pipeline-title">GENAI</div>
        <div class="pipeline-sub">Explanation</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node active">
        <div>◆</div>
        <div class="pipeline-title">RISK ENGINE</div>
        <div class="pipeline-sub">Decision</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-node">
        <div>◎</div>
        <div class="pipeline-title">MODULE 3</div>
        <div class="pipeline-sub">Lending</div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LAYER 1
    # --------------------------------------------------------

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">1</div>
    <div>
        <div class="layer-name">Module 1 · Borrower Input</div>
        <div class="layer-sub">
            Actual input entering the live fraud pipeline
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Borrower Address",
            borrower_address[:18] + "..."
            if borrower_address and len(borrower_address) > 21
            else borrower_address or "—"
        )

    with c2:
        st.metric(
            "DR Penalty",
            f"+{module1_penalty * 100:.2f}%"
        )

    with c3:
        st.metric(
            "Payload Status",
            "RECEIVED"
            if module1_payload
            else "MANUAL INPUT"
        )

    with st.expander("View complete Module 1 payload"):
        st.json(safe(module1_payload))

    # --------------------------------------------------------
    # LAYER 2 — WALLET LOOKUP
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">2</div>
    <div>
        <div class="layer-name">Wallet Lookup · Dataset Match</div>
        <div class="layer-sub">
            Module 2 searches the live dataset using the borrower address
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        if assessment.get("wallet_found"):

            a, b, c = st.columns(3)

            with a:
                st.metric(
                    "Wallet Match",
                    "FOUND"
                )

            with b:
                st.metric(
                    "Matching Records",
                    assessment.get(
                        "match_count",
                        1
                    )
                )

            with c:
                st.metric(
                    "Dataset Row",
                    assessment.get(
                        "dataset_index",
                        "—"
                    )
                )

            st.success(
                "Borrower address successfully matched to "
                "a wallet record in transaction_dataset.csv."
            )

            with st.expander(
                "View matched wallet transaction features"
            ):
                wallet_tx = assessment.get(
                    "wallet_transaction",
                    {}
                )

                st.dataframe(
                    pd.DataFrame(
                        [wallet_tx]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.error(
                "Wallet was not found in the dataset."
            )

    else:
        st.info(
            "Run the live assessment above to perform the wallet lookup."
        )

    # --------------------------------------------------------
    # LAYER 3 — XGBOOST
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">3</div>
    <div>
        <div class="layer-name">XGBoost · Fraud Detection</div>
        <div class="layer-sub">
            Wallet features → imputation → SHAP selection → XGBoost
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        pipeline = assessment.get(
            "feature_pipeline",
            {}
        )

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Original Features",
                pipeline.get(
                    "original_features",
                    "—"
                )
            )

        with b:
            st.metric(
                "After Imputation",
                pipeline.get(
                    "features_after_imputation",
                    "—"
                )
            )

        with c:
            st.metric(
                "SHAP Selected",
                pipeline.get(
                    "selected_features",
                    "—"
                )
            )

        with d:
            st.metric(
                "XGBoost Probability",
                f"{assessment.get('base_xgboost_probability', 0) * 100:.2f}%"
            )

        progression = assessment.get(
            "boosting_progression",
            []
        )

        if progression:
            checkpoints=get_boosting_checkpoints(progression)
            st.markdown("### 🌳 Intermediate XGBoost Tree Analysis")
            st.markdown("Each new boosting tree learns from the errors/residuals of the existing ensemble and adds a correction. The probabilities below are cumulative ensemble outputs, not independent predictions from individual trees.")
            if checkpoints:
                cols=st.columns(len(checkpoints))
                for i,r in enumerate(checkpoints):
                    with cols[i]: st.metric("FINAL" if i==len(checkpoints)-1 else f"TREE {r['iteration']}",f"{r['fraud_probability']*100:.2f}%")
                fig=make_boosting_chart(progression,assessment.get("threshold",0.5))
                if fig: st.pyplot(fig,use_container_width=True); plt.close(fig)
                st.markdown("#### 🔍 How each stage contributes")
                for i,r in enumerate(checkpoints):
                    prev=checkpoints[i-1] if i else None
                    with st.expander(f"Tree {r['iteration']}" + (" — Final" if i==len(checkpoints)-1 else ""),expanded=(i==0)):
                        st.write(checkpoint_contribution_text(r,prev))
                        if prev is not None:
                            a,b,c=st.columns(3)
                            a.metric("Previous",f"{prev['fraud_probability']*100:.2f}%")
                            b.metric("Current",f"{r['fraud_probability']*100:.2f}%")
                            c.metric("Change",f"{r['change']*100:+.2f} pp")
            with st.expander("View raw boosting progression"):
                st.dataframe(pd.DataFrame(normalize_boosting_progression(progression)),use_container_width=True,hide_index=True)

    else:
        st.info(
            "XGBoost results will appear after processing the borrower."
        )

    # --------------------------------------------------------
    # GENAI — USER-FACING EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">✦</div>
    <div>
        <div class="layer-name">GenAI · Simple Explanation</div>
        <div class="layer-sub">
            Gemini converts the strongest model signals into understandable points
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        ai_explanation = assessment.get(
            "genai_explanation"
        )

        if ai_explanation:
            render_genai_explanation(
                ai_explanation
            )
        else:
            st.info(
                "AI explanation is unavailable. "
                "Run the assessment again."
            )

    else:
        st.info(
            "Process a borrower to generate the AI explanation."
        )

    # --------------------------------------------------------
    # LAYER 4 — SHAP
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">4</div>
    <div>
        <div class="layer-name">SHAP · Explainability</div>
        <div class="layer-sub">
            Feature-level explanation of the current borrower prediction
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        local_shap = assessment.get(
            "local_shap",
            []
        )

        if local_shap:

            shap_threshold = float(assessment.get("threshold", 0.5))
            shap_probability = float(assessment.get("base_xgboost_probability", 0.0))

            render_local_shap_explanation(
                local_shap,
                shap_probability,
                shap_threshold
            )

            fig = make_local_shap_chart(
                local_shap
            )

            if fig:
                st.pyplot(
                    fig,
                    use_container_width=True
                )
                plt.close(fig)

            with st.expander(
                "View local SHAP values"
            ):
                st.dataframe(
                    pd.DataFrame(local_shap),
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info(
                "Local SHAP explanation is unavailable."
            )

        try:
            global_response = requests.get(
                f"{API_URL}/api/v1/explainability",
                timeout=30
            )

            if global_response.ok:

                global_rows = global_response.json().get(
                    "features",
                    []
                )

                if global_rows:

                    st.markdown(
                        "#### Global SHAP Feature Importance"
                    )

                    fig = make_global_shap_chart(
                        global_rows
                    )

                    if fig:
                        st.pyplot(
                            fig,
                            use_container_width=True
                        )
                        plt.close(fig)

        except Exception:
            pass

    else:
        st.info(
            "SHAP explanation will appear after processing."
        )

    # --------------------------------------------------------
    # RISK ENGINE
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">5</div>
    <div>
        <div class="layer-name">Risk Engine · Final Decision</div>
        <div class="layer-sub">
            On-chain XGBoost + off-chain financial risk + Module 1 DR penalty
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        final_risk = float(
            assessment.get(
                "final_risk",
                0
            )
        )

        base_probability = float(
            assessment.get(
                "base_xgboost_probability",
                0
            )
        )

        penalty = float(assessment.get("dr_penalty", 0))
        off_chain = float(assessment.get("off_chain_risk", 0))
        weighted_on = float(assessment.get("weighted_on_chain_risk", base_probability * 0.70))
        weighted_off = float(assessment.get("weighted_off_chain_risk", off_chain * 0.30))

        level, decision, routing, color = risk_info(final_risk)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("On-chain XGBoost", f"{base_probability * 100:.2f}%")
        with c2:
            st.metric("Off-chain Risk", f"{off_chain * 100:.2f}%")
        with c3:
            st.metric("DR Penalty", f"+{penalty * 100:.2f}%")
        with c4:
            st.metric("Final Hybrid Risk", f"{final_risk * 100:.2f}%")

        offchain_data = assessment.get("off_chain_input", {})
        st.markdown("### Hybrid risk composition")
        risk_table = pd.DataFrame([
            ["Loan / Income", str(offchain_data.get("loan_to_income_ratio", 0)), float(offchain_data.get("loan_risk", 0) or 0)],
            ["Liabilities / Income", str(offchain_data.get("liability_to_income_ratio", 0)), float(offchain_data.get("liability_risk", 0) or 0)],
            ["Discrepancy Ratio", str(offchain_data.get("discrepancy_ratio", 0)), float(offchain_data.get("discrepancy_risk", 0) or 0)],
            ["Off-chain combined", "—", float(off_chain)],
            ["Weighted on-chain", "70%", float(weighted_on)],
            ["Weighted off-chain", "30%", float(weighted_off)],
            ["DR penalty", "—", float(penalty)],
            ["FINAL RISK", "—", float(final_risk)],
        ], columns=["Component", "Ratio / Weight", "Risk"])

        st.dataframe(risk_table, use_container_width=True, hide_index=True)

        left, right = st.columns(
            [1.25, 1],
            gap="large"
        )

        with left:
            fig = make_risk_meter(
                final_risk
            )
            st.pyplot(
                fig,
                use_container_width=True
            )
            plt.close(fig)

        with right:
            st.markdown(
                f"""
<div class="risk-box">
    <div class="small-label">Security Decision</div>
    <div class="risk-number" style="color:{color};">
        {decision}
    </div>
    <div style="font-weight:800;color:{color};">
        {level} RISK
    </div>
    <div style="color:#8490a5;font-size:12px;margin-top:8px;">
        {assessment.get("description","")}
    </div>
</div>
""",
                unsafe_allow_html=True
            )

        st.code(
            f"On-chain XGBoost risk = {base_probability:.6f}\n"
            f"Off-chain financial risk = {off_chain:.6f}\n"
            f"On-chain contribution = 0.70 × {base_probability:.6f} = {weighted_on:.6f}\n"
            f"Off-chain contribution = 0.30 × {off_chain:.6f} = {weighted_off:.6f}\n"
            f"DR penalty = +{penalty:.6f}\n"
            f"--------------------------------\n"
            f"Final risk = min(1.0, max(0.0, "
            f"{weighted_on:.6f} + {weighted_off:.6f} + {penalty:.6f}))\n"
            f"Final risk = {final_risk:.6f}\n"
            f"Decision = {decision}"
        )

    else:
        st.info(
            "Risk calculation will appear after processing."
        )

    # --------------------------------------------------------
    # MODULE 3
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
<div class="layer-header">
    <div class="layer-number">6</div>
    <div>
        <div class="layer-name">Module 3 · Lending Handoff</div>
        <div class="layer-sub">
            Final risk output is passed to the DeFi lending simulator
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if assessment:

        module3_payload = assessment.get(
            "module3_payload",
            latest_module2_payload
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Risk",
                f"{assessment.get('final_risk', 0) * 100:.2f}%"
            )

        with c2:
            st.metric(
                "Decision",
                assessment.get(
                    "decision",
                    "—"
                )
            )

        with c3:
            st.metric(
                "Routing",
                risk_info(
                    float(
                        assessment.get(
                            "final_risk",
                            0
                        )
                    )
                )[2]
            )

        with st.expander("View Module 3 payload"):
            st.json(
                safe(module3_payload)
            )

        st.success("Module 2 automatically forwarded this result to Module 3 during assessment.")

        if st.button(
            "RESEND RISK RESULT TO MODULE 3",
            type="secondary"
        ):

            with st.spinner(
                "Sending borrower risk assessment to Module 3..."
            ):

                result = forward_to_module3(
                    module3_payload
                )

            if result["success"]:

                st.success(
                    "Module 3 received the risk assessment."
                )

                st.json(
                    result["data"]
                )

            else:

                st.warning(
                    "Module 3 could not be reached. "
                    "The Module 2 assessment is still complete."
                )

                st.json(
                    result["data"]
                )

    else:
        st.info(
            "Module 3 handoff becomes available after assessment."
        )


# ============================================================
# MODULE 1 PAGE
# ============================================================

elif page == "① Module 1 · Payload":

    st.markdown(
        '<div class="section-title">Layer 1 · Module 1 Payload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'This is the source of the borrower address used by Module 2.'
        '</div>',
        unsafe_allow_html=True
    )

    if module1_payload:
        st.json(safe(module1_payload))
    else:
        st.warning(
            "latest_module1_payload.json was not found. "
            "You can enter the borrower address manually above."
        )


# ============================================================
# XGBOOST PAGE
# ============================================================

elif page == "② Wallet → XGBoost":

    st.markdown(
        '<div class="section-title">Layer 3 · Wallet → XGBoost</div>',
        unsafe_allow_html=True
    )

    if assessment:

        pipeline = assessment.get(
            "feature_pipeline",
            {}
        )

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Dataset Features",
                pipeline.get("original_features", "—")
            )

        with b:
            st.metric(
                "Imputed Features",
                pipeline.get(
                    "features_after_imputation",
                    "—"
                )
            )

        with c:
            st.metric(
                "Selected Features",
                pipeline.get("selected_features", "—")
            )

        with d:
            st.metric(
                "Fraud Probability",
                f"{assessment.get('base_xgboost_probability', 0) * 100:.2f}%"
            )

        progression = assessment.get(
            "boosting_progression",
            []
        )

        if progression:
            checkpoints=get_boosting_checkpoints(progression)
            st.markdown("### 🌳 Intermediate XGBoost Tree Analysis")
            st.markdown("Each successive group of trees refines the existing ensemble. The displayed probabilities are cumulative ensemble outputs.")
            if checkpoints:
                cols=st.columns(len(checkpoints))
                for i,r in enumerate(checkpoints):
                    with cols[i]: st.metric("FINAL" if i==len(checkpoints)-1 else f"TREE {r['iteration']}",f"{r['fraud_probability']*100:.2f}%")
                fig=make_boosting_chart(progression,assessment.get("threshold",.5))
                if fig: st.pyplot(fig,use_container_width=True); plt.close(fig)
                st.markdown("#### 🔍 How each stage contributes")
                for i,r in enumerate(checkpoints):
                    prev=checkpoints[i-1] if i else None
                    with st.expander(f"Tree {r['iteration']}" + (" — Final" if i==len(checkpoints)-1 else "")):
                        st.write(checkpoint_contribution_text(r,prev))

    else:
        st.info(
            "Process a borrower first."
        )


# ============================================================
# SHAP PAGE
# ============================================================

elif page == "③ SHAP · Explainability":

    st.markdown(
        '<div class="section-title">Layer 4 · SHAP Explainability</div>',
        unsafe_allow_html=True
    )

    if assessment:

        local = assessment.get(
            "local_shap",
            []
        )

        if local:

            shap_threshold = float(assessment.get("threshold", 0.5))
            shap_probability = float(assessment.get("base_xgboost_probability", 0.0))

            render_local_shap_explanation(
                local,
                shap_probability,
                shap_threshold
            )

            fig = make_local_shap_chart(local)

            if fig:
                st.pyplot(
                    fig,
                    use_container_width=True
                )
                plt.close(fig)

            st.dataframe(
                pd.DataFrame(local),
                use_container_width=True,
                hide_index=True
            )

        try:
            response = requests.get(
                f"{API_URL}/api/v1/explainability",
                timeout=30
            )

            if response.ok:

                rows = response.json().get(
                    "features",
                    []
                )

                if rows:

                    fig = make_global_shap_chart(rows)

                    if fig:
                        st.pyplot(
                            fig,
                            use_container_width=True
                        )
                        plt.close(fig)

        except Exception:
            pass

    else:
        st.info(
            "Process a borrower first."
        )


# ============================================================
# MODULE 3 PAGE
# ============================================================

elif page == "④ Module 3 · Handoff":

    st.markdown(
        '<div class="section-title">Layer 6 · Module 3 Handoff</div>',
        unsafe_allow_html=True
    )

    if assessment:

        payload = assessment.get(
            "module3_payload",
            latest_module2_payload
        )

        st.json(
            safe(payload)
        )

        if st.button(
            "SEND TO MODULE 3",
            type="primary"
        ):

            result = forward_to_module3(
                payload
            )

            if result["success"]:
                st.success(
                    "Module 3 received the assessment."
                )
            else:
                st.error(
                    "Module 3 is unavailable."
                )

            st.json(
                result["data"]
            )

    else:
        st.info(
            "Process a borrower first."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    try:
        response = requests.get(
            f"{API_URL}/api/v1/model",
            timeout=30
        )

        model_info = (
            response.json()
            .get("model", {})
            if response.ok
            else {}
        )

    except Exception:
        model_info = {}

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Model",
            model_info.get(
                "model",
                "XGBoost"
            )
        )

    with c2:
        st.metric(
            "Boosting Rounds",
            model_info.get(
                "boosting_rounds",
                "—"
            )
        )

    with c3:
        st.metric(
            "Original Features",
            model_info.get(
                "original_features",
                "—"
            )
        )

    with c4:
        st.metric(
            "Selected Features",
            model_info.get(
                "selected_features",
                "—"
            )
        )

    st.markdown("---")

    st.info(
        "Training metrics are loaded from the saved model artifacts. "
        "The live demo prediction is generated from the wallet matched "
        "by Module 1's borrower address."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
<div style="display:flex;justify-content:space-between;
            color:#59657a;font-size:11px;padding:5px 0;">
    <div>◈ DeFiLens · Live DeFi Transaction Intelligence</div>
    <div>Module 1 · XGBoost · SHAP · GenAI · Risk Engine · Module 3</div>
</div>
""",
    unsafe_allow_html=True
)
