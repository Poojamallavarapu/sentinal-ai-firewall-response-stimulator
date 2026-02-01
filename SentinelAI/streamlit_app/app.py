import streamlit as st
import requests
import time
import os

API_URL = os.getenv(
    "SENTINEL_API_URL",
    "https://sentinal-ai-firewall-response-stimulator.onrender.com/analyze"
)

st.set_page_config(
    page_title="SentinelAI Firewall",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ SentinelAI – AI Firewall Simulator")
st.subheader("🌐 Network Security Check")

with st.status("Analyzing network traffic...", expanded=True):
    time.sleep(1)

    try:
        response = requests.post(
            API_URL,
            json={"request_type": "network"},
            timeout=20
        )
        response.raise_for_status()
        net_result = response.json()

    except requests.exceptions.RequestException:
        st.error("❌ Backend service is temporarily unavailable.")
        st.info("Please wait a few seconds and refresh.")
        st.stop()

# 🚫 BLOCK
if net_result["decision"] == "BLOCK":
    st.error("🚫 Access blocked due to network policy.")
    st.stop()

# ⚠️ WARNING
if net_result["decision"] == "WARNING":
    st.warning("⚠️ Suspicious activity detected.")
    st.info("Access restricted.")
    st.stop()

# ✅ ALLOW
if net_result["decision"] == "ALLOW":
    st.session_state["network_verified"] = True
    st.success("✅ Network verified successfully.")
    st.info("➡️ Open **Prompt** page from the left sidebar.")
    st.stop()
