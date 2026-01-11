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
import json

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_metadata_map(df):
    """PERCEPTION: Extracts metadata map with safe example values."""
    return {
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "example_values": {col: df[col].dropna().unique()[:3].tolist() for col in df.columns},
        "numeric_summary": df.describe().to_dict()
    }

@app.get("/")
async def health_check():
    return {"status": "Agentic Backend Online"}

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), user_query: str = ""):
    try:
        # 1. LOAD DATA & PERCEIVE
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        meta_map = get_metadata_map(df)
        
        # 2. COMBINED PROMPT (Quota-Saving & Rule-Enforcing)
        combined_prompt = f"""
        User Query: {user_query}
        Metadata Map: {meta_map}
        
        TASK:
        1. Determine if intent is INFO (questions about data) or ACTION (cleaning/plotting).
        2. Provide response in EXACT JSON format.

        STRICT RULES FOR 'ACTION' CONTENT:
        - DATA INTEGRITY: Keep original names (e.g., 'Laptop'). NEVER map to 'A, B, C'.
        - NUMERIC CLEANING: Use MEDIAN for nulls in Age and Quantity to handle outliers.
        - DATA TYPING: Cast Age and Quantity to int after filling: .fillna(0).astype(int).
        - TEXT CLEANING: Standardize 'Region' or text columns to Title Case if inconsistent.
        - LIBRARIES: Use 'df' for data, 'px' for plotly, and save chart to 'fig'.
        - OUTPUT: Provide ONLY valid Python code in the 'content' field.

        STRICT JSON OUTPUT FORMAT:
        {{
            "intent": "INFO" or "ACTION",
            "content": "Your text answer if INFO, or your Python code if ACTION"
        }}
        """

        # Single call to save free-tier quota
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=combined_prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        # 3. PARSE AND EXECUTE
        res_data = json.loads(response.text)
        intent = res_data.get("intent")
        content = res_data.get("content")

        if intent == "INFO":
            return {
                "chat_answer": content,
                "metadata_analysed": meta_map,
                "data_preview": df.head(10).to_dict(orient="records")
            }
        
        # ACTION: Local Execution (Sovereign Layer)
        local_vars = {"df": df, "px": px, "fig": None}
        exec(content, {}, local_vars)
        
        # Capture cleaned dataframe and chart
        res_df = local_vars.get("df")
        res_df = res_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
        
        fig = local_vars.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else None
        
        return {
            "metadata_analysed": meta_map,
            "ai_code": content,
            "fig_json": fig_json,
            "data_preview": res_df.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        if "429" in str(e):
            return {"error": "API quota reached. Please wait 60 seconds."}
        return {"error": str(e), "ai_code": "Error in agent execution."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)