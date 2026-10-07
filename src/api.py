from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}


class StudentData(BaseModel):
    attendance: float = Field(ge = 0, le = 100)
    study_hours: float = Field(ge = 1, le = 8)
    previous_marks: float = Field(ge = 0, le = 100)
    assignment_score: float = Field(ge = 0, le = 100)
    internal_marks: float = Field(ge = 0, le = 100)

@app.post("/predict")
def predict(student: StudentData):

    student_df = pd.DataFrame([{
        "attendance": student.attendance,
        "study_hours": student.study_hours,
        "previous_marks": student.previous_marks,
        "assignment_score": student.assignment_score,
        "internal_marks": student.internal_marks
    }])

    student_scaled = scaler.transform(student_df)

    prediction = model.predict(student_scaled)

    probabilities = model.predict_proba(student_scaled)

    confidence = probabilities[0][prediction[0]]

    if prediction[0] == 1:
        result = "Pass"
    else:
        result = "Fail"

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }