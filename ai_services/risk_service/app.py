import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import os
import time
import threading
import uvicorn
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeFi RiskGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #8b949e;
    font-size: 17px;
    margin-bottom: 30px;
}

.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 15px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

.high-card {
    background: rgba(248, 81, 73, 0.10);
    border: 1px solid rgba(248, 81, 73, 0.40);
    border-radius: 14px;
    padding: 25px;
}

.medium-card {
    background: rgba(210, 153, 34, 0.10);
    border: 1px solid rgba(210, 153, 34, 0.40);
    border-radius: 14px;
    padding: 25px;
}

.low-card {
    background: rgba(63, 185, 80, 0.10);
    border: 1px solid rgba(63, 185, 80, 0.40);
    border-radius: 14px;
    padding: 25px;
}

.recommendation {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #388bfd;
    border-radius: 10px;
    padding: 15px;
    margin: 8px 0;
}

.stButton > button {
    width: 100%;
    font-weight: 700;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/defi_risk_dataset.csv"

TARGET = "risk_level"

FEATURES = [
    "transaction_count",
    "borrow_count",
    "supply_count",
    "repay_count",
    "withdraw_count",
    "liquidation_count",
    "flashloan_count",
    "failed_transaction_count",
    "total_value",
    "average_gas_used",
    "average_gas_price",
    "repayment_ratio",
    "supply_borrow_ratio",
    "withdraw_borrow_ratio",
    "failed_transaction_ratio"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
    else:
        # Fallback synthetic frame if dataset CSV is not found directly
        data = pd.DataFrame(np.random.rand(100, len(FEATURES)), columns=FEATURES)
        data[TARGET] = np.random.choice(["Low", "Medium", "High"], size=100)
    return data


df = load_data()


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    X = X.fillna(X.median())

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Test prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    return model, metrics


model, metrics = train_model(df)

# ============================================================
# INITIALIZE & LOAD INGESTED MODULE 2 PAYLOAD
# ============================================================

default_m2_payload = {
    "borrower_wallet_address": "",
    "verified_monthly_income": 0.0,
    "total_liabilities": 0.0,
    "requested_loan_amount": 0.0,
    "discrepancy_ratio": 0.0,
    "dr_risk_penalty": 0.0,
    "on_chain_risk": 0.0,
    "off_chain_risk": 0.0,
    "final_risk": 0.0,
    "risk_level": "UNKNOWN",
    "decision": "UNKNOWN"
}

# Load the latest payload produced by Module 2.
# Module 3 does not use a hardcoded borrower or risk result.
if os.path.exists("latest_module2_payload.json"):
    try:
        with open("latest_module2_payload.json", "r", encoding="utf-8") as f:
            module2_payload = json.load(f)
    except Exception:
        module2_payload = default_m2_payload.copy()
else:
    module2_payload = default_m2_payload.copy()


# Sync into session state for Streamlit layer rendering
st.session_state["module2_payload"] = module2_payload


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:26px;font-weight:800;">
        🛡️ DeFi RiskGuard
        </div>

        <div style="color:#8b949e;font-size:13px;">
        RISK PREDICTION & PREVENTION
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📥 Layer 1 · Ingested Module 2 Payload",
            "⛓️ Risk Prediction",
            "📊 Model Performance",
            "🔎 Dataset Analysis",
            "🛡️ Prevention"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("<b>Connection Status</b>", unsafe_allow_html=True)
    st.success("🟢 Listening on FastAPI Port 8002")

    st.divider()

    st.caption("Machine Learning")
    st.caption("Random Forest Classifier")
    st.caption("DeFi Wallet Behaviour")


# ============================================================
# HELPER FUNCTION
# ============================================================

def risk_recommendations(risk, values):

    recommendations = []

    if values["liquidation_count"] > 0:
        recommendations.append(
            "Review liquidation activity and maintain adequate collateral."
        )

    if values["repayment_ratio"] < 0.20:
        recommendations.append(
            "Improve repayment consistency and reduce unnecessary borrowing."
        )

    if values["failed_transaction_ratio"] > 0.10:
        recommendations.append(
            "Investigate repeated failed transactions."
        )

    if (
        values["borrow_count"] > 0
        and values["supply_borrow_ratio"] < 0.50
    ):
        recommendations.append(
            "Maintain stronger supply/collateral support relative to borrowing."
        )

    if risk == "High":

        recommendations.extend([
            "Reduce borrowing exposure.",
            "Closely monitor wallet activity.",
            "Review liquidation and repayment behaviour."
        ])

    elif risk == "Medium":

        recommendations.extend([
            "Monitor borrowing and withdrawal behaviour.",
            "Maintain adequate collateral.",
            "Review repayment activity regularly."
        ])

    else:

        recommendations.extend([
            "Continue healthy repayment behaviour.",
            "Monitor collateral levels regularly.",
            "Avoid unnecessary borrowing exposure."
        ])

    return list(dict.fromkeys(recommendations))


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🛡️ Risk Prediction & Prevention</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning based DeFi Financial Risk Analysis Platform</div>',
        unsafe_allow_html=True
    )

    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Features",
            len(FEATURES)
        )

    with c3:
        st.metric(
            "Risk Classes",
            df[TARGET].nunique()
        )

    with c4:
        st.metric(
            "ML Model",
            "Random Forest"
        )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        ⛓️ DeFi Risk Prediction System
        </div>

        The system analyses DeFi wallet behaviour and predicts
        financial risk using Machine Learning.

        <br><br>

        <b>Input:</b> Blockchain wallet behavioural features

        <br>

        <b>Model:</b> Random Forest Classifier

        <br>

        <b>Output:</b> Low / Medium / High Risk

        <br>

        <b>Risk Score:</b> Probability of High Risk × 100

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔄 Prediction Workflow")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="card">
            <b>01 — Wallet Behaviour</b>
            <br><br>
            Collect DeFi transaction activity.
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
            <b>02 — Feature Analysis</b>
            <br><br>
            Analyse borrowing, repayment,
            liquidation and failed transactions.
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
            <b>03 — ML Prediction</b>
            <br><br>
            Random Forest predicts the risk class.
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="card">
            <b>04 — Prevention</b>
            <br><br>
            Generate risk reduction recommendations.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LAYER 1: INGESTED MODULE 2 PAYLOAD DISPLAY
# ============================================================

elif page == "📥 Layer 1 · Ingested Module 2 Payload":

    st.markdown(
        '<div class="main-title">📥 Ingested Module 2 Payload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Displays payload received from Module 2 (Fraud Service) & details forwarded to Module 4 (Blockchain)</div>',
        unsafe_allow_html=True
    )

    if st.button("🔄 Refresh Live Ingested Data"):
        st.rerun()

    c_m2, c_m4 = st.columns(2)

    with c_m2:
        st.markdown("### 📥 **Received from Module 2 (Port 8002)**")
        st.json(module2_payload)

    with c_m4:
        st.markdown("### ⛓️ **Data Prepared for Module 4 (Smart Contracts)**")

        fin_score = module2_payload.get("final_risk_score", 0.0073)
        collateral_req = 1.10 if fin_score < 0.40 else 1.50
        health_factor = round(1.25 / fin_score, 2) if fin_score > 0 else 2.50

        blockchain_payload = {
            "borrower_wallet_address": module2_payload.get("borrower_wallet_address", "0x000..."),
            "approved_loan_amount": module2_payload.get("requested_loan_amount", 0.0),
            "final_risk_score": fin_score,
            "execution_status": module2_payload.get("execution_status", "UNKNOWN"),
            "required_collateral_ratio": collateral_req,
            "simulated_health_factor": health_factor,
            "blockchain_mint_ready": module2_payload.get("authorized_for_module3", False)
        }

        st.json(blockchain_payload)

    st.divider()
    st.markdown("### 🛡️ **Risk & Verification Overview**")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Borrower Wallet", f"{module2_payload.get('borrower_wallet_address', '')[:6]}...")
    with m2: st.metric("Loan Request", f"${module2_payload.get('requested_loan_amount', 0):,.2f}")
    with m3: st.metric("Risk Score", f"{module2_payload.get('final_risk_score', 0)*100:.2f}%")
    with m4: st.metric("Execution Status", module2_payload.get("execution_status", "N/A"))


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "⛓️ Risk Prediction":

    st.markdown(
        '<div class="main-title">⛓️ DeFi Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter wallet behaviour and predict financial risk</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🔐 Wallet Behaviour Input
        </div>

        Enter the behavioural information of a DeFi wallet.
        The Random Forest model will predict Low, Medium or High risk.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_count = st.number_input("Transaction Count", min_value=0, value=10, step=1)
        borrow_count = st.number_input("Borrow Count", min_value=0, value=2, step=1)
        supply_count = st.number_input("Supply Count", min_value=0, value=2, step=1)
        repay_count = st.number_input("Repay Count", min_value=0, value=1, step=1)
        withdraw_count = st.number_input("Withdraw Count", min_value=0, value=1, step=1)

    with col2:
        liquidation_count = st.number_input("Liquidation Count", min_value=0, value=0, step=1)
        flashloan_count = st.number_input("Flashloan Count", min_value=0, value=0, step=1)
        failed_transaction_count = st.number_input("Failed Transaction Count", min_value=0, value=0, step=1)
        total_value = st.number_input("Total Value", min_value=0.0, value=0.0, step=100.0)
        average_gas_used = st.number_input("Average Gas Used", min_value=0.0, value=300000.0, step=10000.0)

    with col3:
        average_gas_price = st.number_input("Average Gas Price", min_value=0.0, value=2.5e10, format="%.2e")
        repayment_ratio = st.number_input("Repayment Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        supply_borrow_ratio = st.number_input("Supply / Borrow Ratio", min_value=0.0, value=1.0, step=0.10)
        withdraw_borrow_ratio = st.number_input("Withdraw / Borrow Ratio", min_value=0.0, value=0.50, step=0.10)
        failed_transaction_ratio = st.number_input("Failed Transaction Ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button("🔍 ANALYSE DEFI RISK", type="primary", use_container_width=True)

    if predict_button:

        input_data = pd.DataFrame([{
            "transaction_count": transaction_count,
            "borrow_count": borrow_count,
            "supply_count": supply_count,
            "repay_count": repay_count,
            "withdraw_count": withdraw_count,
            "liquidation_count": liquidation_count,
            "flashloan_count": flashloan_count,
            "failed_transaction_count": failed_transaction_count,
            "total_value": total_value,
            "average_gas_used": average_gas_used,
            "average_gas_price": average_gas_price,
            "repayment_ratio": repayment_ratio,
            "supply_borrow_ratio": supply_borrow_ratio,
            "withdraw_borrow_ratio": withdraw_borrow_ratio,
            "failed_transaction_ratio": failed_transaction_ratio
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if "High" in classes:
            high_index = classes.index("High")
            high_probability = probabilities[high_index]
        else:
            high_probability = 0.0

        risk_score = high_probability * 100

        st.divider()
        st.markdown("### 🛡️ Risk Assessment Result")

        r1, r2, r3 = st.columns(3)
        with r1: st.metric("Risk Score", f"{risk_score:.2f} / 100")
        with r2: st.metric("Predicted Risk", prediction)
        with r3: st.metric("High Risk Probability", f"{high_probability * 100:.2f}%")

        gauge_col, result_col = st.columns([1.2, 1])

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={"suffix": " / 100", "font": {"size": 30}},
                    title={"text": "DeFi Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {
                            "color": "#f85149" if prediction == "High" else "#e3b341" if prediction == "Medium" else "#3fb950"
                        }
                    }
                )
            )

            fig.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6edf3"},
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        with result_col:
            if prediction == "High":
                st.markdown('<div class="high-card"><h2>🔴 HIGH RISK</h2><b>Immediate review recommended.</b><br><br>The model detected behavioural patterns associated with high DeFi financial risk.</div>', unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown('<div class="medium-card"><h2>🟠 MEDIUM RISK</h2><b>Monitoring recommended.</b><br><br>The model detected moderate-risk behavioural patterns.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="low-card"><h2>🟢 LOW RISK</h2><b>Relatively healthy behaviour.</b><br><br>The model detected relatively healthy DeFi behavioural patterns.</div>', unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown('<div class="main-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Random Forest evaluation results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with c2: st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    with c3: st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    with c4: st.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "🔎 Dataset Analysis":

    st.markdown('<div class="main-title">🔎 Dataset Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analysis of defi_risk_dataset.csv</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Records", f"{len(df):,}")
    with c2: st.metric("Columns", len(df.columns))
    with c3: st.metric("Features", len(FEATURES))
    with c4: st.metric("High Risk Records", int((df[TARGET] == "High").sum()) if TARGET in df.columns else 0)

    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(50), hide_index=True, use_container_width=True, height=400)


# ============================================================
# PREVENTION
# ============================================================

elif page == "🛡️ Prevention":

    st.markdown('<div class="main-title">🛡️ Risk Prevention</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Risk reduction recommendations</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="low-card"><h2>🟢 LOW RISK</h2>✓ Continue healthy repayment<br><br>✓ Monitor collateral levels</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="medium-card"><h2>🟠 MEDIUM RISK</h2>✓ Monitor borrowing activity<br><br>✓ Maintain adequate collateral</div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="high-card"><h2>🔴 HIGH RISK</h2>✓ Reduce borrowing exposure<br><br>✓ Review liquidation activity</div>', unsafe_allow_html=True)


# ============================================================
# FASTAPI BACKGROUND ENDPOINT RECEIVER
# MODULE 3 → MODULE 4
# ============================================================

api_app = FastAPI(
    title="Module 4 Risk Endpoint Receiver"
)


# ============================================================
# MODULE 3 PAYLOAD MODEL
# ============================================================

class BlockchainRiskPayload(BaseModel):

    module: str = "module_3_blockchain"

    timestamp: str = ""

    transaction: dict = {}

    borrower: dict = {}

    loan: dict = {}

    blockchain: dict = {}

    dynamicRisk: dict = {}

    fraud: dict = {}

    financial: dict = {}

    behavior_features: Optional[dict] = None


# ============================================================
# MODULE 3 → MODULE 4
# RECEIVE UPDATED BLOCKCHAIN DATA
# ============================================================
from datetime import datetime, timezone
@api_app.post("/api/risk/update")
def receive_blockchain_update(
    payload: BlockchainRiskPayload
):

    try:

        # Convert received payload to dictionary
        payload_dict = payload.dict()


        # ----------------------------------------------------
        # SAVE LATEST BLOCKCHAIN PAYLOAD
        # ----------------------------------------------------

        with open(
            "latest_blockchain_payload.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload_dict,
                f,
                indent=2
            )


        # ----------------------------------------------------
        # ALSO SAVE THE PAYLOAD IN SESSION STATE
        # ----------------------------------------------------

        st.session_state[
            "blockchain_payload"
        ] = payload_dict


        # ----------------------------------------------------
        # EXTRACT BLOCKCHAIN INFORMATION
        # ----------------------------------------------------

        blockchain =  payload_dict.get(   "blockchain",  {}  )


        platform =  blockchain.get(   "platform",   {})


        borrower = blockchain.get( "borrower",   {})


        transaction =   payload_dict.get(   "transaction", {}  )


        loan =payload_dict.get("loan", {} )


        fraud =payload_dict.get( "fraud", {} )


        financial =payload_dict.get("financial", {})


        dynamic_risk = payload_dict.get( "dynamicRisk", {})


        # ----------------------------------------------------
        # PREPARE RESPONSE
        # ----------------------------------------------------

        response = {

            "status":
                "SUCCESS",

            "message":
                "Blockchain state received successfully by Module 4.",

            "module":
                "module_4_dynamic_risk",

            "transaction":
                transaction,

            "borrower":
                borrower,

            "loan":
                loan,

            "blockchain":
                blockchain,

            "dynamicRisk":
                dynamic_risk,

            "fraud":
                fraud,

            "financial":
                financial,

            "received_at":
                datetime.now( timezone.utc ).isoformat()
        }


        # ----------------------------------------------------
        # LOG RECEIVED DATA
        # ----------------------------------------------------

        print(
            "\n=================================================="
        )

        print(
            "📥 MODULE 4 RECEIVED DATA FROM MODULE 3"
        )

        print(
            "=================================================="
        )

        print(
            "Borrower:",
            borrower.get(
                "wallet",
                "N/A"
            )
        )

        print(
            "Transaction ID:",
            transaction.get(
                "transactionId",
                "N/A"
            )
        )

        print(
            "Transaction Status:",
            transaction.get(
                "status",
                "N/A"
            )
        )

        print(
            "Transaction Hash:",
            transaction.get(
                "txHash",
                "N/A"
            )
        )

        print(
            "Loan Amount:",
            loan.get(
                "requestedAmount",
                0
            )
        )

        print(
            "Collateral:",
            loan.get(
                "depositedCollateral",
                0
            )
        )

        print(
            "Health Ratio:",
            dynamic_risk.get(
                "healthRatioPercent",
                0
            )
        )

        print(
            "Collateral / Loan Ratio:",
            dynamic_risk.get(
                "collateralLoanRatio",
                "N/A"
            )
        )

        print(
            "Liquidity Risk:",
            dynamic_risk.get(
                "liquidityRisk",
                "N/A"
            )
        )

        print(
            "Fraud Probability:",
            fraud.get(
                "probability",
                0
            )
        )

        print(
            "Final Risk:",
            financial.get(
                "finalRisk",
                0
            )
        )

        print(
            "==================================================\n"
        )


        return response


    except Exception as err:

        print(
            "❌ Module 4 API Error:",
            str(err)
        )

        raise HTTPException(

            status_code=500,

            detail=
                f"Module 4 Error: {str(err)}"
        )


# ============================================================
# GET LATEST BLOCKCHAIN DATA
# ============================================================

@api_app.get(
    "/api/risk/latest"
)
def get_latest_blockchain_data():

    try:

        if not os.path.exists(
            "latest_blockchain_payload.json"
        ):

            return {

                "status":
                    "NO_DATA",

                "message":
                    "No blockchain data received from Module 3 yet."
            }


        with open(
            "latest_blockchain_payload.json",
            "r",
            encoding="utf-8"
        ) as f:

            payload = json.load(f)


        return {

            "status":
                "SUCCESS",

            "payload":
                payload
        }


    except Exception as err:

        raise HTTPException(

            status_code=500,

            detail=
                str(err)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@api_app.get(
    "/api/health"
)
def health_check():

    return {

        "status":
            "SUCCESS",

        "module":
            "Module 4",

        "service":
            "Dynamic Risk & Health Prediction",

        "port":
            8002,

        "receiving_from":
            "Module 3",

        "endpoint":
            "/api/risk/update"
    }


# ============================================================
# FASTAPI SERVER
# ============================================================

def run_fastapi_server():

    uvicorn.run(

        api_app,

        host="0.0.0.0",

        port=8002,

        log_level="error"
    )


# ============================================================
# RUN BACKGROUND FASTAPI SERVER
# ============================================================

if not any(
    thread.name == "m3_fastapi_thread"
    for thread in threading.enumerate()
):

    thread = threading.Thread(

        target=
            run_fastapi_server,

        name=
            "m3_fastapi_thread",

        daemon=True
    )

    thread.start()