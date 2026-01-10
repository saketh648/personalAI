import streamlit as st
import requests
import plotly.io as pio

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🤖 AI Data Analyst")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
query = st.text_input("What should I do with this data?")

if uploaded_file is not None:
    if st.button("Run Analysis"):
        files = {"file": uploaded_file.getvalue()}
        params = {"user_query": query}
        
        with st.spinner("AI is thinking and coding..."):
            try:
                res = requests.post("http://127.0.0.1:8000/analyze", files=files, params=params)
                if res.status_code == 200:
                    data = res.json()
                    
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success("Analysis Complete!")
                        
                        # 1. Display the Chart
                        if data.get("fig_json"):
                            fig = pio.from_json(data["fig_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 2. Display the Code
                        with st.expander("View AI Code"):
                            st.code(data["ai_code"], language="python")
                            
                        # 3. Display Data Preview
                        st.write("### Data Preview (Post-Analysis)")
                        st.dataframe(data["data_preview"])
                else:
                    st.error(f"Backend Error: {res.status_code}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
else:
    st.info("Please upload a CSV file to begin.")