from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import AmazonHelpAgent


st.set_page_config(
    page_title="AmazonHelp AI Support Agent",
    page_icon="📦",
    layout="wide",
)

st.title("📦 AmazonHelp AI Support Agent")

st.caption(
    "Intent classification → historical retrieval → grounding → "
    "response generation → escalation"
)

st.info(
    "Demo deployment uses a 500-row temporal-safe historical subset. "
    "Reported evaluation metrics were produced on the full retrieval corpus."
)

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not configured in this deployment."
    )


@st.cache_resource
def load_agent():
    return AmazonHelpAgent(
        retrieval_path=ROOT / "data" / "demo_retrieval.csv"
    )


query = st.text_area(
    "Customer message",
    placeholder="Example: Where is my package?",
    height=120,
)

top_k = st.slider(
    "Historical interactions to retrieve",
    min_value=1,
    max_value=5,
    value=5,
)

run = st.button(
    "Analyze request",
    type="primary",
    use_container_width=True,
)

if run:

    if not query.strip():
        st.error("Please enter a customer message.")
        st.stop()

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "This deployment needs OPENAI_API_KEY to generate a response."
        )
        st.stop()

    try:
        with st.spinner(
            "Classifying, retrieving evidence, and drafting response..."
        ):
            agent = load_agent()

            result = agent.run(
                query=query.strip(),
                top_k=top_k,
                model=os.getenv("OPENAI_MODEL"),
            )

    except Exception as exc:
        st.error(f"Agent error: {exc}")
        st.stop()

    # ------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------

    st.subheader("Intent")

    intent = result["intent"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted intent",
            intent["intent"],
        )

    with c2:
        st.metric(
            "Confidence band",
            intent["confidence_band"],
        )

    with c3:
        st.metric(
            "Semantic score",
            f"{intent['score']:.3f}",
        )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    st.subheader("Historical evidence")

    assessment = result["evidence_assessment"]

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Top similarity",
            f"{assessment['top_score']:.3f}",
        )

    with c2:
        st.metric(
            "Evidence sufficient",
            str(assessment["evidence_sufficient"]),
        )

    for item in result["retrieved_evidence"]:

        with st.expander(
            f"Rank {item['rank']} • "
            f"similarity {item['dense_score']:.3f}"
        ):

            st.markdown("**Historical customer**")
            st.write(
                item["historical_customer_text"]
            )

            if item.get("brand_response"):
                st.markdown("**Historical AmazonHelp response**")
                st.write(
                    item["brand_response"]
                )

    # ------------------------------------------------------------
    # Response
    # ------------------------------------------------------------

    st.subheader("Draft response")

    if result["should_escalate"]:
        st.warning("⚠️ Human escalation recommended")
    else:
        st.success("✅ Suitable for autonomous handling")

    st.write(
        result["draft_response"]
    )

    if result["grounded_claim"]:
        st.caption(
            "Grounded claim: "
            + result["grounded_claim"]
        )

    if result["needs_more_information"]:
        st.info(
            "The agent believes additional customer information "
            "is required."
        )

    # ------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------

    st.subheader("Escalation decision")

    if result["escalation_reasons"]:
        for reason in result["escalation_reasons"]:
            st.write(f"• {reason}")
    else:
        st.write("No escalation triggers.")
