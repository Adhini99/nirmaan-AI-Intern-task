# streamlit app placeholder
# streamlit_app.py
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/score"  # change if deployed

st.title("Spoken Introduction Scorer (Demo)")

transcript = st.text_area("Paste transcript here:", height=200)
if st.button("Score"):
    if not transcript.strip():
        st.warning("Please paste a transcript.")
    else:
        payload = {"transcript": transcript}
        resp = requests.post(API_URL, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            st.metric("Overall score", f"{data['overall_score']} / 100")
            st.write(f"Words: {data['words']}")
            for c in data['per_criterion']:
                st.subheader(f"{c['criterion_name']} — {c['score_out_of_100']} / 100")
                st.write(f"Combined signal: {c['S_combined']} (rule {c['S_rule']}, sem {c['S_sem']}, len {c['S_len']})")
                st.write("Keywords matched:", c['matched_keywords'])
                st.write("Feedback:", c['feedback'])
        else:
            st.error(f"Error scoring: {resp.status_code} - {resp.text}")
