# ⚙️ AI Data Pipeline Architect

An automated ETL generation tool that transforms messy CSV data into production-ready SQL and Python scripts.

## 🚀 Features
- **Schema Inference**: Automatically detects data types for Snowflake/BigQuery.
- **ETL Generation**: Converts natural language into PySpark or dbt models.
- **Data Quality**: Generates Pydantic and Great Expectations test suites.

## 🏗 Project Structure
- `/backend`: FastAPI server handling Gemini AI logic and code generation.
- `/frontend`: Streamlit dashboard for pipeline configuration.

## 🛠 Setup
1. Clone the repo: `git clone ...`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your key to `.env`: `GEMINI_API_KEY=AIza...`
4. Run Backend: `uvicorn backend.main:app --reload`
5. Run Frontend: `streamlit run frontend/app.py`