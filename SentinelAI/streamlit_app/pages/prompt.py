import streamlit as st
import requests
import os

# -----------------------------------
# Protect page (network must be verified)
# -----------------------------------
if "network_verified" not in st.session_state:
    st.error("🚫 Unauthorized access. Please complete network verification first.")
    st.stop()

# -----------------------------------
# Backend API URL
# -----------------------------------
API_URL = os.getenv(
    "SENTINEL_API_URL",
    "http://127.0.0.1:8000/analyze"
)

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="SentinelAI – Prompt Security",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SentinelAI – Prompt Injection Detection")
st.write("This page analyzes prompts using a trained deep‑learning model.")

# -----------------------------------
# Prompt input
# -----------------------------------
prompt = st.text_area(
    "Enter your prompt",
    placeholder="Type your prompt here..."
)

# -----------------------------------
# Analyze button
# -----------------------------------
if st.button("Analyze Prompt"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Analyzing prompt security..."):
            response = requests.post(
                API_URL,
                json={
                    "request_type": "prompt",
                    "data": {"text": prompt}
                }
            )

        res = response.json()

        # -----------------------------------
        # Result display
        # -----------------------------------
        st.subheader("🔍 Analysis Result")

        st.markdown(f"**Decision:** {res['decision']}")
        st.markdown(f"**Confidence Score:** {res['confidence']:.2f}")

        if res["decision"] == "BLOCK":
            st.error("🚫 Malicious prompt detected.")
        elif res["decision"] == "WARNING":
            st.warning("⚠️ Suspicious prompt detected.")
        else:
            st.success("✅ Prompt is safe.")
