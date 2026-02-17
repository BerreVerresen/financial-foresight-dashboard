# Financial Foresight Dashboard (Frontend)

This is the standalone frontend for the AI Financial Foresight system.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Backend Connection

This dashboard requires the **Backend API** to be running.
By default, it connects to: `http://localhost:8000/api/benchmarks`

To change the backend URL (e.g. for production), set the environment variable:
`BACKEND_API_URL=https://your-api-url.com/api/benchmarks`
