import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeFi Lending Finance",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD OPENAI API KEY
# ============================================================

# IMPORTANT:
# Store your OpenAI API key in the environment instead of
# hardcoding it inside the source code.
#
# Windows PowerShell example:
# $env:OPENAI_API_KEY="your-new-api-key"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

elif os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ============================================================
# IMPORT EXTRACTION ENGINE
# ============================================================

from gpt_document_extractor import (
    MODEL_NAME,
    extract_from_documents,
    apply_external_values,
    create_final_output
)


# ============================================================
# API TRANSMISSION HELPER TO MODULE 2
# ============================================================

def send_docai_output_to_fraud_engine(final_output_data):
    """
    Sends Module 1 extracted JSON payload to Module 2
    Fraud Detection API.

    Module 2 FastAPI:
    http://localhost:8001/api/v1/assess-risk
    """

    url = "http://localhost:8001/api/v1/assess-risk"

    # --------------------------------------------------------
    # Extract values from Module 1
    # --------------------------------------------------------

    wallet = (
        final_output_data.get(
            "borrower_wallet_address"
        )
        or ""
    )

    income = float(
        final_output_data.get(
            "verified_monthly_income"
        )
        or 0.0
    )

    liabilities = float(
        final_output_data.get(
            "total_liabilities"
        )
        or 0.0
    )

    requested_loan = float(
        final_output_data.get(
            "requested_loan_amount"
        )
        or 0.0
    )

    collateral = float(
        final_output_data.get(
            "collateral_amount"
        )
        or 0.0
    )

    discrepancy_ratio = float(
        final_output_data.get(
            "discrepancy_ratio"
        )
        or 0.0
    )


    # --------------------------------------------------------
    # Calculate DR risk penalty
    # --------------------------------------------------------

    dr_risk_penalty = (
        0.35
        if discrepancy_ratio > 5.0
        else 0.0
    )


    # --------------------------------------------------------
    # Payload sent to Module 2
    # --------------------------------------------------------

    docai_payload = {

        "borrower_wallet_address":
            wallet,

        "verified_monthly_income":
            income,

        "total_liabilities":
            liabilities,

        "requested_loan_amount":
            requested_loan,

        "collateral_amount":
            collateral,

        "discrepancy_ratio":
            discrepancy_ratio,

        "dr_risk_penalty":
            dr_risk_penalty
    }


    # --------------------------------------------------------
    # Send request
    # --------------------------------------------------------

    try:

        response = requests.post(

            url,

            json=docai_payload,

            timeout=10
        )


        # Raise an exception for HTTP errors such as
        # 404, 422, 500, etc.
        response.raise_for_status()


        return response.json()


    except requests.exceptions.ConnectionError as e:

        st.warning(
            f"Note: Could not reach Module 2 API on Port 8001: {e}"
        )

        return None


    except requests.exceptions.Timeout as e:

        st.warning(
            f"Note: Module 2 API on Port 8001 timed out: {e}"
        )

        return None


    except requests.exceptions.HTTPError as e:

        st.error(
            f"Module 2 API returned an HTTP error: {e}"
        )

        try:

            st.json(
                response.json()
            )

        except Exception:

            st.text(
                response.text
            )

        return None


    except Exception as e:

        st.error(
            f"Module 2 API transmission failed: {e}"
        )

        return None


# ============================================================
# HEADER
# ============================================================

st.title(
    "💰 DeFi Lending Finance"
)

st.subheader(
    "Welcome to DeFi Lending Finance"
)

st.write(
    "Upload your financial document and extract "
    "structured borrower financial information."
)

st.divider()


# ============================================================
# FINANCE DOCUMENT ANALYSIS
# ============================================================

st.header(
    "📄 Finance Document Analysis"
)

st.write(
    "Upload your financial document here and click Analyze."
)


# ============================================================
# IMAGE INPUT ONLY
# ============================================================

uploaded_files = st.file_uploader(

    "Upload Financial Document",

    type=[
        "png",
        "jpg",
        "jpeg"
    ],

    accept_multiple_files=True
)


# ============================================================
# OPTIONAL APPLICATION INPUTS
# ============================================================

st.subheader(
    "Application Information"
)


col1, col2 = st.columns(2)


# ============================================================
# BORROWER WALLET
# ============================================================

with col1:

    wallet_address = st.text_input(

        "Borrower Wallet Address",

        value=(
            "0x00009277775ac7d0d59eaad8fee3d10ac6c805e8"
        ),

        placeholder="0x..."
    )


# ============================================================
# REQUESTED LOAN
# ============================================================

with col2:

    requested_loan = st.number_input(

        "Requested Loan Amount (ETH)",

        min_value=0.0,

        value=2.0,

        step=0.1,

        format="%.4f"
    )


# ============================================================
# COLLATERAL
# ============================================================

collateral_amount = st.number_input(

    "Collateral Amount (ETH)",

    min_value=0.0,

    value=5.0,

    step=0.1,

    format="%.4f"
)


st.caption(
    "DeFi loan and collateral values are represented in ETH."
)


# ============================================================
# COLLATERAL COVERAGE PREVIEW
# ============================================================

if requested_loan > 0:

    collateral_coverage = (
        collateral_amount /
        requested_loan
    ) * 100

    st.info(
        f"Collateral Coverage: "
        f"{collateral_coverage:.2f}%"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(

    "🔍 Analyze",

    type="primary",

    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Validate document
    # --------------------------------------------------------

    if not uploaded_files:

        st.warning(
            "Please upload at least one PNG, JPG or JPEG financial document."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate wallet
    # --------------------------------------------------------

    wallet_address = wallet_address.strip()


    if not wallet_address:

        st.warning(
            "Please provide a borrower wallet address."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate loan amount
    # --------------------------------------------------------

    if requested_loan <= 0:

        st.warning(
            "Requested loan amount must be greater than 0 ETH."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate collateral
    # --------------------------------------------------------

    if collateral_amount <= 0:

        st.warning(
            "Collateral amount must be greater than 0 ETH."
        )

        st.stop()


    # --------------------------------------------------------
    # Check minimum collateral requirement
    # --------------------------------------------------------

    required_collateral = (
        requested_loan * 1.50
    )


    if collateral_amount < required_collateral:

        st.error(

            f"Insufficient collateral. "
            f"Minimum required collateral is "
            f"{required_collateral:.4f} ETH "
            f"for a {requested_loan:.4f} ETH loan."
        )

        st.stop()


    # --------------------------------------------------------
    # Temporary document directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(
            temp_dir
        )

        document_paths = []


        # ----------------------------------------------------
        # Save uploaded images
        # ----------------------------------------------------

        for uploaded_file in uploaded_files:

            file_path = (
                temp_path /
                uploaded_file.name
            )


            with open(
                file_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            document_paths.append(
                file_path
            )


        # ----------------------------------------------------
        # External application inputs
        # ----------------------------------------------------

        external_inputs = {

            "wallet_address":
                wallet_address,

            "requested_loan_amount":
                requested_loan,

            "collateral_amount":
                collateral_amount
        }


        # ----------------------------------------------------
        # Document analysis
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing financial document(s)..."
        ):

            try:

                # --------------------------------------------
                # Extract information
                # --------------------------------------------

                extracted = extract_from_documents(

                        document_paths,

                        external_inputs
                    )


                # --------------------------------------------
                # Apply external application values
                # --------------------------------------------

                extracted = apply_external_values(

                        extracted,

                        external_inputs
                    )


                # --------------------------------------------
                # Create final output
                # --------------------------------------------

                final_output = create_final_output(

                        extracted
                    )


                # ========================================================
                # FORCE APPLICATION VALUES INTO FINAL OUTPUT
                # ========================================================

                # Wallet
                final_output[
                    "borrower_wallet_address"
                ] = wallet_address


                # Requested DeFi loan
                final_output[
                    "requested_loan_amount"
                ] = requested_loan


                # Collateral
                final_output[
                    "collateral_amount"
                ] = collateral_amount


                # Explicit DeFi unit information
                final_output[
                    "loan_asset"
                ] = "ETH"


                final_output[
                    "collateral_asset"
                ] = "ETH"


                # ========================================================
                # API TRANSMISSION TO MODULE 2
                # ========================================================

                api_response =send_docai_output_to_fraud_engine(

                        final_output
                    )


                if api_response:

                    st.toast(
                        "✅ Module 1 data transmitted to Module 2 Fraud Engine!"
                    )


            except Exception as error:

                st.error(
                    "Document analysis failed."
                )

                st.exception(
                    error
                )

                st.stop()


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "Financial document analysis completed."
    )


    st.divider()


    # ========================================================
    # EXTRACTED INFORMATION
    # ========================================================

    st.header(
        "📊 Extracted Financial Information"
    )


    wallet = final_output[
        "borrower_wallet_address"
    ]


    income = final_output[
        "verified_monthly_income"
    ]


    liabilities = final_output[
        "total_liabilities"
    ]


    loan = final_output[
        "requested_loan_amount"
    ]


    collateral = final_output[
        "collateral_amount"
    ]


    discrepancy_ratio = final_output[
        "discrepancy_ratio"
    ]


    # ========================================================
    # WALLET
    # ========================================================

    st.subheader(
        "1. Borrower Wallet Address"
    )


    if wallet:

        st.code(
            wallet
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # MONTHLY INCOME
    # ========================================================

    st.subheader(
        "2. Verified Monthly Income"
    )


    if income is not None:

        st.write(
            f"{income:,.2f}"
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # LIABILITIES
    # ========================================================

    st.subheader(
        "3. Total Liabilities"
    )


    if liabilities is not None:

        st.write(
            f"{liabilities:,.2f}"
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # REQUESTED LOAN
    # ========================================================

    st.subheader(
        "4. Requested Loan Amount"
    )


    if loan is not None:

        st.write(
            f"{loan:.4f} ETH"
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # COLLATERAL
    # ========================================================

    st.subheader(
        "5. Collateral Amount"
    )


    if collateral is not None:

        st.write(
            f"{collateral:.4f} ETH"
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # COLLATERAL COVERAGE
    # ========================================================

    st.subheader(
        "6. Collateral Coverage"
    )


    if loan and loan > 0:

        coverage = (
                collateral /
                loan
            ) * 100


        st.write(
            f"{coverage:.2f}%"
        )

    else:

        st.info(
            "Not available"
        )


    # ========================================================
    # DISCREPANCY RATIO
    # ========================================================

    st.subheader(
        "7. Discrepancy Ratio"
    )


    if discrepancy_ratio is not None:

        st.write(
            f"{discrepancy_ratio:.4f}"
        )

    else:

        st.info(
            "Not available"
        )


    st.divider()


    # ========================================================
    # JSON OUTPUT
    # ========================================================

    st.header(
        "📋 JSON Output"
    )


    json_string = json.dumps(

        final_output,

        indent=4
    )


    st.code(

        json_string,

        language="json"
    )


    st.download_button(

        label="⬇️ Download JSON",

        data=json_string,

        file_name=
            "module1_financial_output.json",

        mime=
            "application/json",

        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    f"DeFi Lending Finance | {MODEL_NAME}"
)