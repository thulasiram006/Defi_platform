import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_service import get_model_service


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="DeFiLens Fraud Detection API",
    description=(
        "Live wallet fraud detection and DeFi transaction "
        "risk assessment API powered by XGBoost."
    ),
    version="3.0.0",
)

model_service = get_model_service()


# ============================================================
# SERVICE URLS
# ============================================================

# Optional old gateway endpoint.
NODE_GATEWAY_URL = "http://localhost:5000/api/fraud-evaluation"

# Module 3 = Blockchain Module
#
# IMPORTANT:
# Change this URL if your Blockchain Module uses a
# different port or endpoint.
BLOCKCHAIN_MODULE_URL = (
    "http://localhost:5000/api/blockchain/process-transaction"
)


# ============================================================
# LOCAL CACHE
# ============================================================

LATEST_MODULE2_PATH = (
    Path(__file__).resolve().parent
    / "latest_module2_payload.json"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class RiskAssessmentRequest(BaseModel):

    # --------------------------------------------------------
    # Wallet supplied by Module 1
    # --------------------------------------------------------

    borrower_wallet_address: Optional[str] = None
    borrower_address: Optional[str] = None
    wallet_address: Optional[str] = None

    # --------------------------------------------------------
    # Financial information supplied by Module 1
    # --------------------------------------------------------

    verified_monthly_income: float = 0.0
    total_liabilities: float = 0.0
    requested_loan_amount: float = 0.0
    discrepancy_ratio: float = 0.0
    dr_risk_penalty: float = 0.0

    # --------------------------------------------------------
    # Optional direct transaction data
    # --------------------------------------------------------

    transaction: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional direct transaction feature dictionary."
    )

    transaction_hash: Optional[str] = None

    # --------------------------------------------------------
    # Model options
    # --------------------------------------------------------

    include_shap: bool = True
    include_boosting: bool = True

    # --------------------------------------------------------
    # Forwarding options
    # --------------------------------------------------------

    # Optional old gateway forwarding.
    forward_to_gateway: bool = False

    # Blockchain forwarding.
    #
    # IMPORTANT:
    # Even if this is True, the Blockchain Module will
    # ONLY be called when the fraud decision is APPROVED.
    forward_to_blockchain: bool = True

    # Compatibility with Module 2 Streamlit app.
    forward_to_module3: bool = True

    # DeFi transaction information supplied by Module 1.
    action: Optional[str] = "borrow"
    collateral_amount: float = 0.0
    amount: Optional[float] = None


class GatewayRequest(BaseModel):

    assessment: Dict[str, Any]

    wallet_address: Optional[str] = None
    borrower_address: Optional[str] = None
    transaction_hash: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def current_timestamp():
    """
    Return the current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def send_to_service(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 30
):
    """
    Send JSON data to another backend service using HTTP POST.
    """

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )

        try:
            data = response.json()

        except Exception:

            data = {
                "raw_response": response.text
            }

        return {
            "success": response.ok,
            "status_code": response.status_code,
            "response": data
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "status_code": None,
            "response": {
                "error": str(e)
            }
        }


def extract_borrower_address(
    request: RiskAssessmentRequest
):
    """
    Extract the wallet address supplied by Module 1.
    """

    return (
        request.borrower_wallet_address
        or request.borrower_address
        or request.wallet_address
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/v1/health")
def health():

    return {
        "status": "healthy",
        "service": "DeFiLens Fraud Detection Engine",
        "timestamp": current_timestamp(),
        "model": model_service.get_model_info()
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/v1/model")
def model_information():

    return {
        "success": True,
        "model": model_service.get_model_info()
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

@app.get("/api/v1/explainability")
def explainability():

    return {
        "success": True,
        "features": model_service.get_shap_data()
    }


# ============================================================
# LIVE WALLET FRAUD ASSESSMENT
# ============================================================

@app.post("/api/v1/assess-risk")
@app.post("/api/v1/module1-input")
def assess_risk(
    request: RiskAssessmentRequest
):

    # ========================================================
    # STEP 1
    # Receive wallet from Module 1
    # ========================================================

    borrower_address = extract_borrower_address(request)

    # --------------------------------------------------------
    # Store Module 1 input locally for dashboard/debugging
    # --------------------------------------------------------

    try:

        module1_received_path = (
            Path(__file__).resolve().parent
            / "latest_module1_payload.json"
        )

        with open(
            module1_received_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                request.dict(),
                file,
                indent=2
            )

    except Exception as cache_error:

        print(
            "Warning: could not save Module 1 payload: "
            f"{cache_error}"
        )

    # ========================================================
    # STEP 2
    # Run Fraud Detection
    # ========================================================

    # --------------------------------------------------------
    # LIVE WALLET MODE
    # Module 1 supplies borrower wallet address.
    # --------------------------------------------------------

    if borrower_address:

        try:

            assessment = model_service.assess_wallet(

                borrower_address=borrower_address,

                verified_monthly_income=(
                    request.verified_monthly_income
                ),

                total_liabilities=(
                    request.total_liabilities
                ),

                requested_loan_amount=(
                    request.requested_loan_amount
                ),

                discrepancy_ratio=(
                    request.discrepancy_ratio
                ),

                dr_penalty=(
                    request.dr_risk_penalty
                ),

                include_shap=(
                    request.include_shap
                ),

                include_boosting=(
                    request.include_boosting
                )
            )

        except Exception as e:

            import traceback

            print("\n" + "=" * 70)
            print("DEFILENS WALLET FRAUD ASSESSMENT ERROR")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70 + "\n")

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

        # ----------------------------------------------------
        # Wallet not found
        # ----------------------------------------------------

        if not assessment.get(
            "wallet_found",
            False
        ):

            raise HTTPException(
                status_code=404,
                detail=(
                    "Borrower wallet was not found in "
                    "transaction_dataset.csv."
                )
            )

    # ========================================================
    # BACKWARD COMPATIBILITY
    # Direct transaction mode
    # ========================================================

    elif request.transaction:

        try:

            assessment = model_service.assess(

                transaction=request.transaction,

                dr_penalty=(
                    request.dr_risk_penalty
                )
            )

        except Exception as e:

            import traceback

            print("\n" + "=" * 70)
            print("DEFILENS TRANSACTION ASSESSMENT ERROR")
            print("=" * 70)

            traceback.print_exc()

            print("=" * 70 + "\n")

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide borrower_address from Module 1 "
                "or direct transaction data."
            )
        )

    # ========================================================
    # STEP 3
    # Add metadata to fraud assessment
    # ========================================================

    assessment["timestamp"] = current_timestamp()

    assessment["borrower_address"] = (
        borrower_address
    )

    assessment["wallet_address"] = (
        request.wallet_address
        or borrower_address
    )

    assessment["transaction_hash"] = (
        request.transaction_hash
    )

    # ========================================================
    # STEP 4
    # Extract Fraud Decision
    # ========================================================

    decision = assessment.get(
        "decision"
    )

    risk_level = assessment.get(
        "risk_level"
    )

    fraud_probability = assessment.get(
        "on_chain_risk"
    )

    # --------------------------------------------------------
    # Normalize decision for reliable comparison
    # --------------------------------------------------------

    if isinstance(decision, str):

        decision_normalized = (
            decision.strip()
            .upper()
        )

    else:

        decision_normalized = ""


    # ========================================================
    # STEP 5
    # Create Blockchain Module Payload
    # ========================================================

    blockchain_payload = {

        # ----------------------------------------------------
        # Borrower information
        # ----------------------------------------------------

        "borrower_wallet_address": (
            borrower_address
        ),

        "wallet_address": (
            request.wallet_address
            or borrower_address
        ),

        # ----------------------------------------------------
        # Financial information from Module 1
        # ----------------------------------------------------

        "verified_monthly_income": (
            request.verified_monthly_income
        ),

        "total_liabilities": (
            request.total_liabilities
        ),

        "requested_loan_amount": (
            request.requested_loan_amount
        ),

        "discrepancy_ratio": (
            request.discrepancy_ratio
        ),

        "dr_risk_penalty": (
            assessment.get(
                "dr_penalty",
                request.dr_risk_penalty
            )
        ),

        # ----------------------------------------------------
        # DeFi transaction information from Module 1
        # ----------------------------------------------------

        "action": (
            request.action
            or "borrow"
        ),

        "amount": (
            request.amount
            if request.amount is not None
            else request.requested_loan_amount
        ),

        "collateral_amount": (
            request.collateral_amount
        ),

        "collateral": (
            request.collateral_amount
        ),

        # ----------------------------------------------------
        # Fraud detection results from Module 2
        # ----------------------------------------------------

        "fraud_probability": (
            fraud_probability
        ),

        "fraud_status": (
            risk_level
        ),

        "decision": (
            decision
        ),

        "risk_level": (
            risk_level
        ),

        "final_risk": (
            assessment.get("final_risk", 0.0)
        ),

        "on_chain_risk": (
            assessment.get("on_chain_risk", 0.0)
        ),

        "off_chain_risk": (
            assessment.get("off_chain_risk", 0.0)
        ),

        "authorized_for_module3": (
            decision_normalized == "APPROVED"
        ),

        # ----------------------------------------------------
        # Transaction information
        # ----------------------------------------------------

        "transaction_hash": (
            request.transaction_hash
        ),

        "timestamp": (
            current_timestamp()
        )
    }


    # ========================================================
    # STEP 6
    # Save Module 2 result locally
    # ========================================================

    try:

        with open(
            LATEST_MODULE2_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                blockchain_payload,
                file,
                indent=2
            )

    except Exception as cache_error:

        print(
            "Warning: could not save Module 2 payload: "
            f"{cache_error}"
        )


    # ========================================================
    # STEP 7
    # SEND TO BLOCKCHAIN ONLY IF APPROVED
    # ========================================================

    blockchain_result = None

    if (
        (
            request.forward_to_blockchain
            or request.forward_to_module3
        )
        and decision_normalized == "APPROVED"
    ):

        print("\n" + "=" * 70)
        print("DEFILENS → BLOCKCHAIN MODULE")
        print("=" * 70)

        print(
            f"Wallet: {borrower_address}"
        )

        print(
            f"Fraud Probability: {fraud_probability}"
        )

        print(
            f"Risk Level: {risk_level}"
        )

        print(
            f"Decision: {decision}"
        )

        print(
            f"Sending to: {BLOCKCHAIN_MODULE_URL}"
        )

        print("=" * 70 + "\n")


        blockchain_result = send_to_service(

            BLOCKCHAIN_MODULE_URL,

            blockchain_payload,

            timeout=30
        )


    # ========================================================
    # STEP 8
    # REJECTED TRANSACTION
    # ========================================================

    else:

        if decision_normalized != "APPROVED":

            print("\n" + "=" * 70)
            print("DEFILENS TRANSACTION REJECTED")
            print("=" * 70)

            print(
                f"Wallet: {borrower_address}"
            )

            print(
                f"Fraud Probability: "
                f"{fraud_probability}"
            )

            print(
                f"Risk Level: {risk_level}"
            )

            print(
                f"Decision: {decision}"
            )

            print(
                "Blockchain Module was NOT called."
            )

            print(
                "No on-chain write will be performed."
            )

            print("=" * 70 + "\n")


    # ========================================================
    # OPTIONAL OLD GATEWAY
    # ========================================================

    gateway_result = None

    if request.forward_to_gateway:

        gateway_result = send_to_service(

            NODE_GATEWAY_URL,

            {
                "wallet_address": (
                    request.wallet_address
                    or borrower_address
                ),

                "transaction_hash": (
                    request.transaction_hash
                ),

                "assessment": assessment
            }
        )


    # Make the exact Module 3 payload available to the
    # Streamlit dashboard through the existing assessment object.
    assessment["module3_payload"] = blockchain_payload

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

        "service": (
            "DeFiLens Hybrid Fraud Detection Engine"
        ),

        # ----------------------------------------------------
        # Fraud assessment
        # ----------------------------------------------------

        "assessment": assessment,

        # ----------------------------------------------------
        # Blockchain payload
        # ----------------------------------------------------

        "blockchain_payload": (
            blockchain_payload
        ),

        # ----------------------------------------------------
        # Blockchain response
        # ----------------------------------------------------

        "blockchain": (
            blockchain_result
        ),

        # ----------------------------------------------------
        # Status information
        # ----------------------------------------------------

        "blockchain_forwarded": (
            blockchain_result is not None
        ),

        "blockchain_forward_reason": (

            "Transaction approved by fraud detection"
            if blockchain_result is not None
            else
            "Transaction was not approved by fraud detection"
        ),

        # ----------------------------------------------------
        # Optional gateway
        # ----------------------------------------------------

        "gateway": gateway_result
    }


# ============================================================
# SIMPLE PREDICTION
# ============================================================

@app.post("/api/v1/predict")
def predict(
    request: RiskAssessmentRequest
):

    borrower_address = extract_borrower_address(
        request
    )

    # --------------------------------------------------------
    # Wallet prediction
    # --------------------------------------------------------

    if borrower_address:

        try:

            result = model_service.assess_wallet(

                borrower_address=borrower_address,

                verified_monthly_income=(
                    request.verified_monthly_income
                ),

                total_liabilities=(
                    request.total_liabilities
                ),

                requested_loan_amount=(
                    request.requested_loan_amount
                ),

                discrepancy_ratio=(
                    request.discrepancy_ratio
                ),

                dr_penalty=(
                    request.dr_risk_penalty
                ),

                include_shap=False,

                include_boosting=False
            )

            if not result.get(
                "wallet_found"
            ):

                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Borrower wallet not found "
                        "in dataset."
                    )
                )

            return {

                "success": True,

                "prediction": result
            }

        except HTTPException:

            raise

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


    # --------------------------------------------------------
    # Direct transaction prediction
    # --------------------------------------------------------

    if not request.transaction:

        raise HTTPException(
            status_code=400,
            detail=(
                "Transaction data cannot be empty."
            )
        )

    try:

        result = model_service.predict(
            request.transaction
        )

        return {

            "success": True,

            "prediction": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GATEWAY-FRIENDLY ENDPOINT
# ============================================================

@app.post("/api/fraud-evaluation")
def fraud_evaluation(
    request: GatewayRequest
):

    assessment = request.assessment

    return {

        "success": True,

        "wallet_address": (
            request.wallet_address
            or request.borrower_address
        ),

        "transaction_hash": (
            request.transaction_hash
        ),

        "risk": (
            assessment.get("final_risk")
        ),

        "risk_level": (
            assessment.get("risk_level")
        ),

        "decision": (
            assessment.get("decision")
        ),

        "action": (
            assessment.get("action")
        ),

        "timestamp": current_timestamp()
    }