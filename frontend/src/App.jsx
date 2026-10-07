import { useState } from "react";
import "./App.css";

function App() {

  const [formData, setFormData] = useState({
    attendance: "",
    study_hours: "",
    previous_marks: "",
    assignment_score: "",
    internal_marks: ""
  });

  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");
  const [confidence, setConfidence] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    for (let field in formData) {
      if (formData[field] === "") {
        setError("Please fill in all fields.");
        return;
      }
    }

    setError("");
    setPrediction("");

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error("Something went wrong");
      }

      const data = await response.json();

      setPrediction(data.prediction);
      setConfidence(data.confidence);

    } catch (error) {
      setError("Unable to connect to the prediction server.");
    }
  };

  const handleReset = () => {
    setFormData({
      attendance: "",
      study_hours: "",
      previous_marks: "",
      assignment_score: "",
      internal_marks: ""
    });

    setPrediction("");
    setError("");
    setConfidence("");
  };

  return (
    <div className="app">

      <div className="main-card">

        <div className="hero">
          <div className="hero-badge">
            MACHINE LEARNING
          </div>

          <h1>Student Performance</h1>

          <p>
            Analyze student academic information and predict
            their expected performance.
          </p>
        </div>

        <div className="content">

          <div className="section-heading">
            <h2>Student Information</h2>
            <p>Enter the student's academic details below.</p>
          </div>

          <form onSubmit={handleSubmit}>

            <div className="form-grid">

              <div className="form-group">
                <label>Attendance</label>

                <div className="input-wrapper">
                  <input
                    type="number"
                    name="attendance"
                    value={formData.attendance}
                    onChange={handleChange}
                    placeholder="82"
                    min="0"
                    max="100"
                  />
                  <span>%</span>
                </div>
              </div>

              <div className="form-group">
                <label>Study Hours</label>

                <div className="input-wrapper">
                  <input
                    type="number"
                    name="study_hours"
                    value={formData.study_hours}
                    onChange={handleChange}
                    placeholder="4.5"
                    min="1"
                    max="8"
                    step="0.1"
                  />
                  <span>hrs</span>
                </div>
              </div>

              <div className="form-group">
                <label>Previous Marks</label>

                <div className="input-wrapper">
                  <input
                    type="number"
                    name="previous_marks"
                    value={formData.previous_marks}
                    onChange={handleChange}
                    placeholder="75"
                    min="0"
                    max="100"
                  />
                  <span>/ 100</span>
                </div>
              </div>

              <div className="form-group">
                <label>Assignment Score</label>

                <div className="input-wrapper">
                  <input
                    type="number"
                    name="assignment_score"
                    value={formData.assignment_score}
                    onChange={handleChange}
                    placeholder="80"
                    min="0"
                    max="100"
                  />
                  <span>/ 100</span>
                </div>
              </div>

              <div className="form-group full-width">
                <label>Internal Marks</label>

                <div className="input-wrapper">
                  <input
                    type="number"
                    name="internal_marks"
                    value={formData.internal_marks}
                    onChange={handleChange}
                    placeholder="78"
                    min="0"
                    max="100"
                  />
                  <span>/ 100</span>
                </div>
              </div>

            </div>

            <div className="button-group">

              <button type="submit">
                Predict Performance
                <span>→</span>
              </button>

              <button
                type="button"
                className="reset-button"
                onClick={handleReset}
              >
                Reset
              </button>

            </div>

          </form>

          {prediction && (
            <div className={`result ${prediction.toLowerCase()}`}>

              <div>
                <p>MODEL PREDICTION</p>
                <h2>{prediction}</h2>
                <span>Confidence: {confidence}%</span>
              </div>

              <div className="result-mark">
                {prediction === "Pass" ? "✓" : "!"}
              </div>

            </div>
          )}

          {error && (
            <div className="error">
              {error}
            </div>
          )}

        </div>

        <div className="footer">
          <span>Student Performance Prediction System</span>
          <span>ML Model • FastAPI • React</span>
        </div>

      </div>

    </div>
  );
}

export default App;