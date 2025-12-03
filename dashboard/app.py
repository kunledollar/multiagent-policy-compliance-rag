import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Policy & Compliance Assistant — RAG Dashboard",
    layout="wide",
)

st.title("Enterprise Policy & Compliance Assistant")
st.caption("Multi-Agent RAG System — Monitoring & Query UI")

with st.sidebar:
    st.header("Backend")
    health = None
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        health = resp.json().get("status")
    except Exception:
        health = "unreachable"
    st.metric("API Health", health)

    if st.button("Trigger Ingestion"):
        with st.spinner("Ingesting policy documents from /data/raw ..."):
            resp = requests.post(f"{API_URL}/ingest", json={})
        st.write(resp.json())

st.subheader("Ask a Compliance Question")

query = st.text_area("Enter your question about policy or compliance", height=120)

if st.button("Run Query") and query.strip():
    with st.spinner("Querying multi-agent RAG backend..."):
        resp = requests.post(f"{API_URL}/query", json={"query": query})
        data = resp.json()

    st.markdown("### Answer")
    st.write(data.get("answer", ""))

    with st.expander("Internal Reasoning"):
        st.write(data.get("reasoning", ""))

    with st.expander("Fact Check Verdict"):
        st.write(data.get("fact_check", ""))

    with st.expander("Retrieved Policy Contexts"):
        for c in data.get("contexts", []):
            st.markdown(f"**Source:** {c.get('source')} (chunk {c.get('chunk_id')})")
            st.write(c.get("text"))
            st.markdown("---")
else:
    st.info("Enter a question and click 'Run Query'.")
