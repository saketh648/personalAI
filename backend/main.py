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
from dotenv import load_dotenv
# 1. Setup & Config
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Provide the key directly as a string
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...), user_query: str = ""):
    try:
        # 2. Load Data
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 3. Instruction for Gemini
        prompt = f"""
        DataFrame columns: {list(df.columns)}
        Task: {user_query}
        
        Rules:
        - Use pandas for cleaning.
        - Use plotly.express (as px) for charts.
        - Assume the dataframe is 'df'.
        - Store the final chart in a variable named 'fig'.
        - DO NOT use fig.show().
        - Provide ONLY the Python code block.
        """
        
        # 4. Generate Code
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        clean_code = response.text.replace("```python", "").replace("```", "").strip()
        
        # 5. Execution Engine (Sandbox)
        local_vars = {"df": df, "px": px, "fig": None}
        exec(clean_code, {}, local_vars)
        
        # 6. JSON Safety Cleaning
        # Convert NaN and Inf to None (null) so the API doesn't crash
        res_df = local_vars.get("df")
        res_df = res_df.replace([np.inf, -np.inf], np.nan).replace({np.nan: None})
        
        # Capture chart if it exists
        fig = local_vars.get("fig")
        fig_json = pio.to_json(fig) if fig is not None else None
        
        return {
            "ai_code": clean_code,
            "fig_json": fig_json,
            "data_preview": res_df.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        return {"error": str(e), "ai_code": "Error during generation or execution"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)