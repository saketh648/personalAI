from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from google import genai
import io
import os
import json
from dotenv import load_dotenv
import plotly.io as pio
import plotly.express as px

# 1. Setup & Initialization
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
    """PERCEPTION: Extracts structural metadata and safe samples."""
    # Convert numeric summary to JSON-safe format immediately
    summary = df.describe().replace({np.nan: None}).to_dict()
    return {
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "example_values": {col: df[col].dropna().unique()[:3].tolist() for col in df.columns},
        "numeric_summary": summary
    }

@app.get("/")
async def health_check():
    return {"status": "Agentic Backend Online", "version": "2.0-JSON-Safe"}

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), user_query: str = ""):
    try:
        # 2. LOAD DATA
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        meta_map = get_metadata_map(df)
        
        # 3. COMBINED PROMPT: Router + Agent in one trip (Saves Quota)
        combined_prompt = f"""
        User Query: {user_query}
        Metadata Map: {meta_map}
        
        TASK:
        1. Determine if intent is INFO (questions) or ACTION (cleaning/plotting).
        2. Provide response in EXACT JSON format.

        STRICT RULES FOR 'ACTION' CONTENT:
        - DATA INTEGRITY: Keep original names (e.g., 'Laptop'). NEVER map to 'A, B, C'.
        - NUMERIC CLEANING: Use MEDIAN for nulls in Age and Quantity to handle outliers.
        - DATA TYPING: Cast Age and Quantity to int after filling: .fillna(0).astype(int).
        - TEXT CLEANING: Standardize 'Region' or text columns to Title Case.
        - LIBRARIES: Use 'df' for data, 'px' for plotly, and save chart to 'fig'.
        - OUTPUT: Provide ONLY valid Python code in the 'content' field.

        STRICT JSON OUTPUT FORMAT:
        {{
            "intent": "INFO" or "ACTION",
            "content": "text answer if INFO, or python code if ACTION"
        }}
        """

        # 4. COGNITION: Single call to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=combined_prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        # Parse the JSON response
        res_data = json.loads(response.text)
        intent = res_data.get("intent")
        content = res_data.get("content")

        # 5. ACTION: Branching Logic
        if intent == "INFO":
            return {
                "chat_answer": content,
                "metadata_analysed": meta_map,
                "data_preview": df.head(10).replace({np.nan: None}).to_dict(orient="records")
            }
        
        # ACTION BRANCH: Local Code Execution
        local_vars = {"df": df, "px": px, "fig": None}
        exec(content, {}, local_vars)
        
        # CRITICAL: Prepare DataFrame for JSON (Replace NaN/Inf with None)
        res_df = local_vars.get("df")
        res_df_safe = res_df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(res_df), None)
        
        fig = local_vars.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else None
        
        return {
            "metadata_analysed": meta_map,
            "ai_code": content,
            "fig_json": fig_json,
            "data_preview": res_df_safe.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        # Handle Rate Limits (429) gracefully
        if "429" in str(e):
            return {"error": "API quota reached. Please wait 60 seconds."}
        return {"error": str(e), "ai_code": "Agent logic failed."}

if __name__ == "__main__":
    import uvicorn
    # Render binding
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)