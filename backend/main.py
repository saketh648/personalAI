from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from google import genai
import io
import os
from dotenv import load_dotenv
import plotly.io as pio
import plotly.express as px

# 1. Setup & Config
load_dotenv()
app = FastAPI()

# Security: Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Streamlit URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_metadata_map(df):
    """
    PERCEPTION MODULE: 
    Extracts structural and statistical metadata without raw values.
    This ensures data sovereignty as raw PII never leaves this server.
    """
    return {
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "shape": df.shape,
        "numeric_summary": df.describe().to_dict() # High-level stats, not raw rows
    }

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), user_query: str = ""):
    try:
        # 2. LOAD DATA (Local only)
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 3. PERCEPTION: Extract metadata map
        meta_map = get_metadata_map(df)
        
        # 4. COGNITIVE: Agentic Reasoning
        # We only send the metadata to the AI, keeping raw data private.
        agent_prompt = f"""
        You are a Senior Data Engineering Agent.
        
        USER QUERY: {user_query}
        
        METADATA MAP (Your only view of the data):
        {meta_map}
        
        TASK:
        1. Identify data quality issues based on null_counts and data_types.
        2. Plan a cleaning strategy (e.g., fill nulls in numeric columns with mean).
        3. Write Python code to execute the plan and answer the user query.
        
        RULES:
        - DATA INTEGRITY: Do not rename, encode, or map categorical values (e.g., keep 'Laptop', do not change to 'A').
        - CLEANING: Only address missing values (NaN). Use fillna with mean/median for numbers and 'Unknown' or mode for text.
        - CONSISTENCY: If the 'Region' column has mixed casing (e.g., 'north' vs 'North'), standardize it to Title Case.
        - LIBRARIES: Use 'df' as the variable, plotly.express as 'px', and store the chart in 'fig'.
        - OUTPUT: Provide ONLY the Python code block (no prose or explanations).
        """
        
        # Call Gemini to generate the "Action" code
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=agent_prompt
        )
        clean_code = response.text.replace("```python", "").replace("```", "").strip()
        
        # 5. ACTION: Local Execution (Sovereign Layer)
        # We execute the AI's logic on the actual data locally.
        local_vars = {"df": df, "px": px, "fig": None}
        exec(clean_code, {}, local_vars)
        
        # Capture transformed data and visualization
        res_df = local_vars.get("df")
        
        # JSON Safety: Convert NaN/Inf to None (null) so the API doesn't crash
        res_df = res_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
        
        fig = local_vars.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else None
        
        return {
            "metadata_analysed": meta_map, # Transparency: Show what the AI saw
            "ai_code": clean_code,
            "fig_json": fig_json,
            "data_preview": res_df.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        return {"error": str(e), "ai_code": "Agent failed to generate or execute logic."}

# Render Deployment Config
if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 and the PORT provided by Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)