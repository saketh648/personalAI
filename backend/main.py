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
    """PERCEPTION: Extracts metadata and small safe samples to guide the AI."""
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
        # 1. LOAD DATA
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        meta_map = get_metadata_map(df)
        
        # 2. INTENT ROUTING: Determine if user wants info or action
        router_prompt = f"""
        User Query: {user_query}
        Metadata: {meta_map}
        
        Is this a question ABOUT the data (INFO) or a request to DO SOMETHING to the data (ACTION)?
        Reply with ONLY 'INFO' or 'ACTION'.
        """
        router_res = client.models.generate_content(model="gemini-2.5-flash", contents=router_prompt)
        intent = router_res.text.strip().upper()

        # --- BRANCH A: DATA INFO (Chat Answer) ---
        if "INFO" in intent and "ACTION" not in intent:
            chat_prompt = f"""
            User asks: {user_query}
            Based ONLY on this metadata: {meta_map}, answer the user's question. 
            Be concise and professional.
            """
            chat_res = client.models.generate_content(model="gemini-2.0-flash", contents=chat_prompt)
            return {
                "chat_answer": chat_res.text,
                "ai_code": "# Info query - no execution needed",
                "metadata_analysed": meta_map,
                "data_preview": df.head(10).to_dict(orient="records")
            }

        # --- BRANCH B: DATA ACTION (Code Execution) ---
        agent_prompt = f"""
        You are a Senior Data Engineering Agent.
        USER QUERY: {user_query}
        METADATA MAP: {meta_map}
        
        TASK: Generate Python code to clean and visualize the data.
        
        STRICT RULES:
        1. DATA INTEGRITY: Keep original names (e.g., 'Laptop'). NEVER map to 'A, B, C'.
        2. CLEANING: Use MEDIAN for numeric nulls (Age, Quantity) to avoid decimals.
        3. TYPING: After filling nulls, cast 'Age' and 'Quantity' to int: .fillna(0).astype(int).
        4. CASING: Standardize 'Region' or text columns to Title Case if inconsistent.
        5. LIBRARIES: Use 'df', plotly.express as 'px', and store the chart in 'fig'.
        6. OUTPUT: Provide ONLY the Python code block.
        """
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=agent_prompt)
        clean_code = response.text.replace("```python", "").replace("```", "").strip()
        
        # LOCAL EXECUTION
        local_vars = {"df": df, "px": px, "fig": None}
        exec(clean_code, {}, local_vars)
        
        res_df = local_vars.get("df")
        res_df = res_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
        
        fig = local_vars.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else None
        
        return {
            "metadata_analysed": meta_map,
            "ai_code": clean_code,
            "fig_json": fig_json,
            "data_preview": res_df.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        return {"error": str(e), "ai_code": "Error in agent routing or execution."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)