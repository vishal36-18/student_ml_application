import pandas as pd
import joblib

model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

print("Model loaded successfully!")


def predict_student(attendance, study_hours, previous_marks,
                    assignment_score, internal_marks):

    new_student = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score,
        "internal_marks": internal_marks
    }])

    new_student_scaled = scaler.transform(new_student)

    prediction = model.predict(new_student_scaled)

    if prediction[0] == 1:
        return "Pass"
    else:
        return "Fail"


def get_number(message, minimum, maximum):

    while True:
        try:
            value = float(input(message))

            if minimum <= value <= maximum:
                return value

            print(f"Please enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")




def main():

    attendance = get_number(
        "Enter attendance (0-100): ",
        0,
        100
    )

    study_hours = get_number(
        "Enter study hours (1-8): ",
        1,
        8
    )

    previous_marks = get_number(
        "Enter previous marks (0-100): ",
        0,
        100
    )

    assignment_score = get_number(
        "Enter assignment score (0-100): ",
        0,
        100
    )

    internal_marks = get_number(
        "Enter internal marks (0-100): ",
        0,
        100
    )

    result = predict_student(
        attendance,
        study_hours,
        previous_marks,
        assignment_score,
        internal_marks
    )

    print("Student Prediction:", result)


if __name__ == "__main__":
    main()