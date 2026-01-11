import streamlit as st
import requests
import plotly.io as pio
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic AI Data Analyst", layout="wide")
st.title("🤖 Agentic AI Data Analyst")

# Get Backend URL
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
query = st.text_input("What should the Agent do with this data?")

if uploaded_file is not None:
    if st.button("Run Agentic Analysis"):
        files = {"file": uploaded_file.getvalue()}
        params = {"user_query": query}
        
        with st.spinner("Agent is perceiving metadata and planning action..."):
            try:
                res = requests.post(f"{BACKEND_URL}/analyze", files=files, params=params)
                
                if res.status_code == 200:
                    data = res.json()
                    
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success("Agentic Process Complete!")

                        # 1. PERCEPTION: Display the Metadata Map
                        # This builds trust by showing exactly what the AI saw (and didn't see)
                        with st.expander("👁️ View Agent's Perception (Metadata Map)"):
                            st.json(data.get("metadata_analysed")) #

                        # 2. ACTION: Display the Chart
                        if data.get("fig_json"):
                            fig = pio.from_json(data["fig_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 3. TRANSPARENCY: Display the Agent's Code
                        with st.expander("📜 View Agent's Generated Logic"):
                            st.code(data["ai_code"], language="python")
                            
                        # 4. RESULTS: Display Data Preview
                        st.write("### Data Preview (Post-Agent Transformation)")
                        st.dataframe(data["data_preview"], use_container_width=True) #
                else:
                    st.error(f"Backend Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")