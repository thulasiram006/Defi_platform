"""
DeFiLens — Gemini GenAI Explanation Engine

IMPORTANT:
- XGBoost makes the fraud/benign prediction.
- SHAP provides the important prediction signals.
- Gemini ONLY converts those signals into simple language.
- Gemini never makes or changes the fraud decision.
"""

import json
import os
import re

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# FEATURE NAME CONVERSION
# ============================================================

FEATURE_NAME_MAP = {

    "total erc20 tnxs":
        "ERC-20 transaction activity",

    "time diff between first and last (mins)":
        "wallet activity duration",

    "unique received from addresses":
        "number of different sending addresses",

    "unique sent to addresses":
        "number of different receiving addresses",

    "erc20 max val rec":
        "largest ERC-20 amount received",

    "erc20 total ether received":
        "total Ether received through ERC-20 activity",

    "avg val received":
        "average amount received",

    "received tnx":
        "number of received transactions",

    "sent tnx":
        "number of sent transactions",

    "total ether received":
        "total Ether received",

    "total ether sent":
        "total Ether sent",

    "total ether balance":
        "Ether balance",

    "min value received":
        "smallest amount received",

    "min val sent":
        "smallest amount sent",

    "erc20 min val rec":
        "smallest ERC-20 amount received",

    "erc20 min val sent":
        "smallest ERC-20 amount sent",

    "erc20 avg val rec":
        "average ERC-20 amount received",

    "erc20 total ether sent":
        "total Ether sent through ERC-20 activity",

    "erc20 uniq rec contract addr":
        "number of ERC-20 receiving contracts",

    "erc20 uniq sent token name":
        "number of different ERC-20 tokens sent",

    "erc20 uniq rec token name":
        "number of different ERC-20 tokens received",
}


def simple_feature_name(feature):
    """
    Convert technical dataset feature names into
    understandable descriptions.
    """

    if not feature:
        return "transaction behaviour"

    key = str(feature).strip().lower()

    if key in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[key]

    # Generic fallback
    readable = str(feature)
    readable = readable.replace("_", " ")
    readable = readable.replace("-", " ")
    readable = re.sub(r"\s+", " ", readable)

    return readable.strip()


# ============================================================
# EXTRACT IMPORTANT SHAP SIGNALS
# ============================================================

def get_relevant_signals(local_shap, fraud_prediction, limit=4):
    """
    For fraud:
        Positive SHAP values are the important suspicious signals.

    For benign:
        Negative SHAP values are the important safe/legitimate signals.

    SHAP is NOT shown to the user.
    It is only used to identify the signals Gemini should explain.
    """

    if not isinstance(local_shap, list):
        return []

    signals = []

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

        if fraud_prediction:

            # Positive SHAP contribution
            # pushes prediction toward fraud.
            if shap_value <= 0:
                continue

            direction = "suspicious"

        else:

            # Negative SHAP contribution
            # pushes prediction toward legitimate.
            if shap_value >= 0:
                continue

            direction = "safe"

        signals.append(
            {
                "feature": simple_feature_name(feature),
                "direction": direction,
                "strength": round(abs(shap_value), 6)
            }
        )

    # Strongest signals first
    signals.sort(
        key=lambda item: item["strength"],
        reverse=True
    )

    return signals[:limit]


# ============================================================
# FALLBACK EXPLANATION
# ============================================================

def fallback_explanation(
    signals,
    fraud_probability,
    fraud_prediction
):
    """
    Used if Gemini is unavailable.

    This guarantees that the Overview page still works
    even when the API is unavailable or rate-limited.
    """

    probability_text = (
        f"{fraud_probability * 100:.2f}%"
    )

    if fraud_prediction:

        title = "Why is this transaction suspicious?"

        points = []

        for signal in signals:

            points.append(
                f"The wallet shows {signal['feature']}, "
                "which increased the model's suspicion."
            )

        if not points:

            points.append(
                "The wallet shows transaction behaviour "
                "associated with higher fraud risk."
            )

        points.append(
            f"The combined signals resulted in a "
            f"{probability_text} fraud probability."
        )

    else:

        title = "Why is this transaction considered safe?"

        points = []

        for signal in signals:

            points.append(
                f"The wallet shows {signal['feature']}, "
                "which supported a legitimate classification."
            )

        if not points:

            points.append(
                "The wallet does not show strong "
                "suspicious transaction patterns."
            )

        points.append(
            f"The combined signals resulted in a relatively "
            f"low {probability_text} fraud probability."
        )

    return {
        "title": title,
        "points": points[:4],
        "source": "Fallback explanation"
    }


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# ============================================================
# GEMINI EXPLANATION
# ============================================================

def generate_gemini_explanation(
    local_shap,
    fraud_probability,
    fraud_prediction
):
    """
    Generate a simple point-by-point explanation using Gemini.

    The prediction itself comes from XGBoost.
    """

    signals = get_relevant_signals(
        local_shap=local_shap,
        fraud_prediction=fraud_prediction,
        limit=4
    )

    # --------------------------------------------------------
    # Always have a fallback.
    # --------------------------------------------------------

    fallback = fallback_explanation(
        signals=signals,
        fraud_probability=fraud_probability,
        fraud_prediction=fraud_prediction
    )

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    client = get_gemini_client()

    if client is None:

        return fallback

    # --------------------------------------------------------
    # Prepare safe signal data.
    #
    # We deliberately do NOT send:
    # - wallet address
    # - income
    # - liabilities
    # - loan amount
    # - raw transaction data
    #
    # Gemini only receives model signals.
    # --------------------------------------------------------

    safe_signals = [
        {
            "signal": item["feature"],
            "direction": item["direction"]
        }
        for item in signals
    ]

    signals_json = json.dumps(
        safe_signals,
        indent=2
    )

    prediction_text = (
        "FRAUD / SUSPICIOUS"
        if fraud_prediction
        else "BENIGN / LIKELY LEGITIMATE"
    )

    if fraud_prediction:

        title = "Why is this transaction suspicious?"

        instruction = """
Explain ONLY why the transaction appears suspicious.

Focus on the supplied suspicious signals.

Do not say that the transaction is definitely fraudulent.
Use wording such as:
- suspicious
- unusual
- increased the risk
- contributed to the fraud score
"""

    else:

        title = "Why is this transaction considered safe?"

        instruction = """
Explain ONLY why the transaction appears safe or likely legitimate.

Focus on the supplied safe signals.

Do not invent suspicious behaviour.
Do not claim that the transaction is guaranteed to be safe.
Use wording such as:
- normal
- consistent
- supported a legitimate classification
- reduced suspicion
"""

    prompt = f"""
You are the explanation component of a DeFi fraud detection system.

The XGBoost model has ALREADY made the prediction.

Prediction:
{prediction_text}

Fraud probability:
{fraud_probability * 100:.2f}%

Important model signals:
{signals_json}

{instruction}

Rules:

1. XGBoost is the decision-maker.
2. You are ONLY explaining the result.
3. Do not change the prediction.
4. Do not invent information.
5. Do not mention SHAP.
6. Do not mention machine-learning feature names.
7. Do not mention model weights or mathematical calculations.
8. Use very simple language suitable for a normal user.
9. Generate exactly 3 or 4 short points.
10. Each point must be one sentence.
11. Do not number the points.
12. Return ONLY the points, one per line.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=220,
            )
        )

        text = (
            response.text
            if response and response.text
            else ""
        )

        text = text.strip()

        if not text:

            return fallback

        # ----------------------------------------------------
        # Clean Gemini output.
        # ----------------------------------------------------

        raw_lines = text.splitlines()

        points = []

        for line in raw_lines:

            line = line.strip()

            if not line:
                continue

            # Remove markdown bullets
            line = re.sub(
                r"^[-•*]\s*",
                "",
                line
            )

            # Remove numbering such as:
            # 1.
            # 1)
            line = re.sub(
                r"^\d+[\.\)]\s*",
                "",
                line
            )

            line = line.strip()

            if len(line) < 15:
                continue

            points.append(line)

        points = points[:4]

        if not points:

            return fallback

        return {
            "title": title,
            "points": points,
            "source": "Gemini GenAI"
        }

    except Exception as error:

        print(
            "Gemini explanation unavailable:",
            error
        )

        return fallback


# ============================================================
# STREAMLIT DISPLAY
# ============================================================

def render_genai_explanation(
    st,
    explanation
):
    """
    Render the already-generated explanation.

    This function DOES NOT call Gemini.
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

    if title.startswith("Why is this transaction suspicious?"):

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
    padding:13px 16px;
    margin:8px 0;
    border-radius:10px;
    border:1px solid #202b3d;
    background:#0e131e;
">
    <span style="
        display:inline-block;
        min-width:25px;
        font-weight:800;
    ">
        {index}.
    </span>
    <span>{point}</span>
</div>
""",
            unsafe_allow_html=True
        )

    st.caption(
        f"Explanation source: {source}. "
        "XGBoost remains responsible for the fraud/benign decision."
    )