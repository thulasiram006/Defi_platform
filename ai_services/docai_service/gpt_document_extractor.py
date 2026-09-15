import os
import json
import base64
import argparse
from pathlib import Path

from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "module1_gpt_output.json"


# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "Set your OpenAI API key before running the program."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".txt": "text/plain",
    ".pdf": "application/pdf"
}


# ============================================================
# MODULE 1 OUTPUT SCHEMA
# ============================================================

MODULE1_SCHEMA = {
    "type": "object",
    "properties": {
        "borrower_wallet_address": {
            "type": ["string", "null"]
        },
        "verified_monthly_income": {
            "type": ["number", "null"]
        },
        "total_liabilities": {
            "type": ["number", "null"]
        },
        "requested_loan_amount": {
            "type": ["number", "null"]
        },
        "discrepancy_ratio": {
            "type": ["number", "null"]
        }
    },
    "required": [
        "borrower_wallet_address",
        "verified_monthly_income",
        "total_liabilities",
        "requested_loan_amount",
        "discrepancy_ratio"
    ],
    "additionalProperties": False
}


# ============================================================
# READ IMAGE AND CONVERT TO DATA URL
# ============================================================
import base64
from pathlib import Path

def image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime_type = SUPPORTED_EXTENSIONS.get(ext, "application/octet-stream")

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# FIND DOCUMENTS
# ============================================================

def find_documents():

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found:\n{DOCUMENTS_DIR}"
        )

    documents = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            documents.append(file_path)

    return documents


# ============================================================
# EXTERNAL APPLICATION INPUTS
# ============================================================

def get_external_inputs():

    parser = argparse.ArgumentParser(
        description="Financial Document Analyzer - Module 1"
    )

    parser.add_argument(
        "--wallet-address",
        type=str,
        default=None,
        help="Borrower's public Ethereum wallet address"
    )

    parser.add_argument(
        "--requested-loan",
        type=float,
        default=None,
        help="Requested loan amount"
    )

    args = parser.parse_args()

    return {
        "wallet_address": args.wallet_address,
        "requested_loan_amount": args.requested_loan
    }


# ============================================================
# EXTRACTION INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are a financial document extraction system.

Analyze ALL uploaded financial documents together and extract
the required borrower financial information.

The final output contains exactly five fields:

1. borrower_wallet_address
2. verified_monthly_income
3. total_liabilities
4. requested_loan_amount
5. discrepancy_ratio

IMPORTANT RULES:

GENERAL:

- Never invent information.
- Never guess missing values.
- If a field is not sufficiently supported by the documents,
  return null.
- Treat all uploaded documents as belonging to the same
  borrower unless the documents clearly indicate otherwise.
- Documents may contain unrelated or distractor information.
- Use the information that is relevant to the borrower.

------------------------------------------------------------
1. BORROWER WALLET ADDRESS
------------------------------------------------------------

Extract the borrower's public blockchain wallet address if
explicitly present.

For Ethereum wallets, look for addresses beginning with 0x.

Examples of useful labels:

- Wallet Address
- Blockchain Wallet
- Ethereum Address
- Borrower Wallet
- Public Wallet Address

Do not confuse:

- transaction hash
- sender address
- receiver address
- contract address

with the borrower's wallet address unless the document clearly
identifies it as the borrower's wallet.

If unavailable, return null.

------------------------------------------------------------
2. VERIFIED MONTHLY INCOME
------------------------------------------------------------

Extract the borrower's verified NET monthly income.

Prefer explicit values such as:

- Net Monthly Income
- Monthly Net Income
- Net Income
- Net Pay
- Take Home Pay
- Monthly Salary

If a payslip contains:

Gross Salary
Deductions
Net Salary

use the NET salary/pay as the verified monthly income.

Do NOT automatically treat:

- gross salary
- annual salary
- purchase amounts
- invoice totals
- account balances
- random bank credits

as monthly income.

If annual income is explicitly provided and clearly represents
the borrower's income, it may be converted to monthly income by
dividing by 12.

Do not make unsupported assumptions.

------------------------------------------------------------
3. TOTAL LIABILITIES
------------------------------------------------------------

Extract the borrower's total existing liabilities/debt only
when explicitly supported.

Look for labels such as:

- Total Liabilities
- Total Outstanding Liabilities
- Outstanding Debt
- Total Debt
- Existing Loans
- Outstanding Loan Balance
- Credit Card Outstanding
- Financial Obligations

If the document explicitly provides a "Total Liabilities"
value, use that value.

If individual liabilities are listed but no total is provided,
you may sum clearly identified outstanding liability amounts
when they unambiguously belong to the borrower.

Do NOT classify ordinary purchases as liabilities.

For example:

Amazon Purchase = ₹2,000

does NOT automatically mean:

Total Liabilities = ₹2,000

Likewise, shopping invoices, receipts, rent payments, grocery
purchases, and ordinary transactions must not be treated as
liabilities unless the document explicitly identifies them as
debt or liability.

If liabilities cannot be reliably determined, return null.

------------------------------------------------------------
4. REQUESTED LOAN AMOUNT
------------------------------------------------------------

Extract the loan amount that the borrower is requesting.

Look for labels such as:

- Requested Loan Amount
- Loan Amount Requested
- Requested Amount
- Loan Application Amount
- Amount Requested

Do NOT confuse requested loan amount with:

- existing loan balance
- total liabilities
- purchase amount
- invoice total
- bank balance
- transaction amount

Only use an amount when the document clearly identifies it as
the requested loan amount.

If unavailable, return null.

------------------------------------------------------------
5. DISCREPANCY RATIO
------------------------------------------------------------

Calculate:

discrepancy_ratio =
requested_loan_amount / verified_monthly_income

Only calculate this value when BOTH:

- verified_monthly_income is available
- requested_loan_amount is available

If either value is missing, return null.

If verified_monthly_income is zero or invalid, return null.

------------------------------------------------------------
IMPORTANT SOURCE PRIORITY
------------------------------------------------------------

If external application values are supplied separately for:

- borrower wallet address
- requested loan amount

those external values are authoritative and should override
the corresponding document-extracted values.

The monthly income and total liabilities should primarily come
from the uploaded financial documents.

------------------------------------------------------------
FINAL OUTPUT
------------------------------------------------------------

Return ONLY the five requested fields.

Do NOT return:

- dr_risk_penalty
- wallet_age_days
- fraud_probability
- fraud_prediction
- risk_score
- additional fields

The final JSON must contain exactly:

borrower_wallet_address
verified_monthly_income
total_liabilities
requested_loan_amount
discrepancy_ratio
"""


# ============================================================
# EXTRACT INFORMATION FROM ALL DOCUMENTS
# ============================================================

def extract_from_documents(
    documents,
    external_inputs
):

    content = [
        {
            "type": "input_text",
            "text": SYSTEM_PROMPT
        }
    ]

    # --------------------------------------------------------
    # External application information
    # --------------------------------------------------------

    external_context = """
External application information:

"""

    if external_inputs["wallet_address"] is not None:

        external_context += (
            "Borrower wallet address supplied by application: "
            f"{external_inputs['wallet_address']}\n"
        )

    else:

        external_context += (
            "Borrower wallet address supplied by application: None\n"
        )

    if external_inputs["requested_loan_amount"] is not None:

        external_context += (
            "Requested loan amount supplied by application: "
            f"{external_inputs['requested_loan_amount']}\n"
        )

    else:

        external_context += (
            "Requested loan amount supplied by application: None\n"
        )

    content.append(
        {
            "type": "input_text",
            "text": external_context
        }
    )

    # --------------------------------------------------------
    # Add every uploaded document
    # --------------------------------------------------------

    for document in documents:
        path_obj = Path(document) if isinstance(document, (str, Path)) else Path(document.name)
        ext = path_obj.suffix.lower()

        content.append(
            {
                "type": "input_text",
                "text": f"Financial document: {path_obj.name}"
            }
        )

        if ext == ".txt":
            # For plain text files, read content directly into a text block
            with open(path_obj, "r", encoding="utf-8") as f:
                text_content = f.read()
            content.append(
                {
                    "type": "input_text",
                    "text": f"Document Text Content:\n{text_content}"
                }
            )
        else:
            # For images and PDFs, pass data URL payload
            data_url = image_to_data_url(path_obj)
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": "high"
                }
            )

    # --------------------------------------------------------
    # GPT request
    # --------------------------------------------------------

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "module1_financial_extraction",
                "strict": True,
                "schema": MODULE1_SCHEMA
            }
        }
    )

    raw_output = response.output_text.strip()

    if not raw_output:
        raise RuntimeError(
            "Model returned an empty response."
        )

    try:

        extracted = json.loads(raw_output)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Model returned invalid JSON:\n"
            f"{raw_output}"
        ) from error

    return extracted


# ============================================================
# APPLY EXTERNAL APPLICATION VALUES
# ============================================================

def apply_external_values(
    extracted,
    external_inputs
):

    # External wallet address takes priority
    if external_inputs["wallet_address"] is not None:

        extracted["borrower_wallet_address"] = (
            external_inputs["wallet_address"]
        )

    # External requested loan takes priority
    if external_inputs["requested_loan_amount"] is not None:

        extracted["requested_loan_amount"] = (
            external_inputs["requested_loan_amount"]
        )

    return extracted


# ============================================================
# CALCULATE DISCREPANCY RATIO
# ============================================================

def calculate_discrepancy_ratio(
    extracted
):

    income = extracted.get(
        "verified_monthly_income"
    )

    loan_amount = extracted.get(
        "requested_loan_amount"
    )

    if income is None:
        return None

    if loan_amount is None:
        return None

    try:

        income = float(income)
        loan_amount = float(loan_amount)

    except (TypeError, ValueError):

        return None

    if income <= 0:
        return None

    return loan_amount / income


# ============================================================
# FINAL MODULE 1 OUTPUT
# ============================================================

def create_final_output(
    extracted
):

    discrepancy_ratio = calculate_discrepancy_ratio(
        extracted
    )

    final_output = {
        "borrower_wallet_address": extracted.get(
            "borrower_wallet_address"
        ),
        "verified_monthly_income": extracted.get(
            "verified_monthly_income"
        ),
        "total_liabilities": extracted.get(
            "total_liabilities"
        ),
        "requested_loan_amount": extracted.get(
            "requested_loan_amount"
        ),
        "discrepancy_ratio": discrepancy_ratio
    }

    return final_output


# ============================================================
# SAVE JSON
# ============================================================

def save_output(
    final_output
):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_output,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINANCIAL DOCUMENT ANALYZER - MODULE 1")
    print("=" * 60)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Documents folder: {DOCUMENTS_DIR}"
    )

    print()

    # --------------------------------------------------------
    # External inputs
    # --------------------------------------------------------

    external_inputs = get_external_inputs()

    print("External inputs:")

    print(
        "Wallet address: "
        f"{external_inputs['wallet_address']}"
    )

    print(
        "Requested loan: "
        f"{external_inputs['requested_loan_amount']}"
    )

    print()

    # --------------------------------------------------------
    # Find documents
    # --------------------------------------------------------

    documents = find_documents()

    if not documents:

        raise RuntimeError(
            f"No supported images found in:\n"
            f"{DOCUMENTS_DIR}"
        )

    print(
        f"Documents found: {len(documents)}"
    )

    for document in documents:

        print(
            f"- {document.name}"
        )

    print()

    # --------------------------------------------------------
    # GPT extraction
    # --------------------------------------------------------

    print(
        "Sending documents to GPT..."
    )

    print()

    extracted = extract_from_documents(
        documents,
        external_inputs
    )

    # --------------------------------------------------------
    # Apply external values
    # --------------------------------------------------------

    extracted = apply_external_values(
        extracted,
        external_inputs
    )

    # --------------------------------------------------------
    # Create final 5-field JSON
    # --------------------------------------------------------

    final_output = create_final_output(
        extracted
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        final_output
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("=" * 60)
    print("FINAL MODULE 1 JSON")
    print("=" * 60)

    print(
        json.dumps(
            final_output,
            indent=4
        )
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()