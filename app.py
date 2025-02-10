from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from starlette.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Load trained AI model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = Path("templates/index.html")
    return html_path.read_text(encoding="utf-8")

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Home Route (Displays Form)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Form Submission Route (Handles Loan Prediction)
@app.post("/", response_class=HTMLResponse)
async def predict_loan(
    request: Request,
    income: float = Form(...),
    credit_score: int = Form(...),
    loan_amount: float = Form(...),
    loan_term: int = Form(...),
    debt_to_income: float = Form(...)
):
    # Prepare input data
    input_data = np.array([[income, credit_score, loan_amount, loan_term, debt_to_income]])
    prediction = model.predict(input_data)
    result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"

    # Render the same form with the result
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

