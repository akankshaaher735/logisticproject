"""
GenAI Learning Insight - a Flask app that serves your logistic.pkl model.

How it works (read top to bottom):
  1. Load logistic.pkl once when the app starts.
  2. Read the feature names (and their ORDER) straight from the model, so the
     form and the prediction can never drift out of sync with your training data.
  3. /predict validates the numbers, builds a one-row table in the right column
     order, and returns the prediction plus probabilities.

Your model is only loaded, never retrained or modified.
"""
import logging
import math
import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic.pkl"

# The model only stores class names (High / Low / Medium), not what they measure.
# Rename this to whatever your target column means, e.g. "Predicted impact level".
TARGET_LABEL = "Predicted level"

# Display text for each feature: (label, group, hint, is_encoded).
# The keys are the exact feature names stored inside logistic.pkl.
# "is_encoded" marks text columns the model saw as numbers (the pickle does not
# contain the text -> number mapping, so the form asks for the number).
FIELD_INFO = {
    "Major_Category": ("Major category", "Student profile",
                       "Field of study, entered as the numeric code used when the model was trained.", True),
    "Year_of_Study": ("Year of study", "Student profile",
                      "The student's current year of study.", False),
    "Pre_Semester_GPA": ("Pre-semester GPA", "Student profile",
                         "GPA at the start of the semester.", False),
    "Weekly_GenAI_Hours": ("Weekly GenAI hours", "AI usage",
                           "Hours per week spent using generative AI tools.", False),
    "Primary_Use_Case": ("Primary use case", "AI usage",
                         "What the student mainly uses AI for, as the numeric code from training.", True),
    "Prompt_Engineering_Skill": ("Prompt engineering skill", "AI usage",
                                 "Skill at writing prompts, on the same scale as your training data.", False),
    "Paid_Subscription": ("Paid subscription", "AI usage",
                          "Paid AI plan, entered as the value used in training (for example 0 or 1 if that is how you encoded it).", True),
    "Traditional_Study_Hours": ("Traditional study hours", "Habits and context",
                                "Hours per week of traditional study.", False),
    "Perceived_AI_Dependency": ("Perceived AI dependency", "Habits and context",
                                "How dependent the student feels on AI, on the scale used in training.", False),
    "Institutional_Policy": ("Institutional policy", "Habits and context",
                             "The institution's AI policy, as the numeric code from training.", True),
    "Anxiety_Level_During_Exams": ("Exam anxiety level", "Habits and context",
                                   "Anxiety during exams, on the scale used in training.", False),
    "Skill_Retention_Score": ("Skill retention score", "Habits and context",
                              "Skill retention score, on the scale used in training.", False),
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("insight")
app = Flask(__name__)


def load_model(path):
    """Load a saved model. joblib is tried first, then plain pickle."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---- Load the model once at startup ---------------------------------------
model, FEATURES, MODEL_ERROR = None, [], None
try:
    model = load_model(MODEL_PATH)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("The model does not store its feature names, so the input order cannot be verified.")
    FEATURES = [str(n) for n in names]
    log.info("Model loaded: %s with %d features", type(model).__name__, len(FEATURES))
except Exception as exc:  # keep the site up and show a clear message instead of crashing
    MODEL_ERROR = f"Could not load logistic.pkl: {exc}"
    log.exception(MODEL_ERROR)

SUPPORTS_PROBA = hasattr(model, "predict_proba")
CLASSES = [str(c) for c in getattr(model, "classes_", [])]


def build_groups():
    """Group the form fields for the template, keeping the model's feature order."""
    groups = {}
    for name in FEATURES:
        label, group, hint, encoded = FIELD_INFO.get(
            name, (name.replace("_", " "), "Other inputs", "Numeric value for this feature.", False))
        groups.setdefault(group, []).append(
            {"name": name, "label": label, "hint": hint, "encoded": encoded})
    return groups


def parse_inputs(payload):
    """Check every feature is present and a finite number. Returns (values, errors)."""
    values, errors = {}, {}
    for name in FEATURES:
        raw = payload.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors[name] = "This field is required."
            continue
        if isinstance(raw, bool):
            errors[name] = "Enter a number."
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue
        if not math.isfinite(number):
            errors[name] = "Enter a finite number."
            continue
        values[name] = number
    return values, errors


@app.route("/")
def index():
    config = {
        "features": FEATURES,
        "classes": CLASSES,
        "targetLabel": TARGET_LABEL,
        "supportsProba": SUPPORTS_PROBA,
    }
    return render_template(
        "index.html",
        groups=build_groups(),
        config=config,
        classes=CLASSES,
        target_label=TARGET_LABEL,
        model_error=MODEL_ERROR,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error=MODEL_ERROR or "The model is not available."), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the inputs as a JSON object."), 400

    values, errors = parse_inputs(payload)
    if errors:
        return jsonify(error="Please fix the highlighted fields.", field_errors=errors), 400

    try:
        # Columns follow the model's own feature order, whatever order the user typed in.
        frame = pd.DataFrame([values], columns=FEATURES)
        prediction = str(model.predict(frame)[0])
        result = {"prediction": prediction, "probabilities": None, "confidence": None}

        # Probabilities only appear if the model really provides them.
        if SUPPORTS_PROBA:
            probs = model.predict_proba(frame)[0]
            result["probabilities"] = [
                {"label": c, "value": round(float(p), 4)} for c, p in zip(CLASSES, probs)
            ]
            result["confidence"] = round(float(max(probs)), 4)
        return jsonify(result)
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="The model could not process these values. Check them and try again."), 500


@app.route("/health")
def health():
    """Used by Render to check the app is alive."""
    return jsonify(status="ok" if model is not None else "model_error"), (200 if model is not None else 503)


if __name__ == "__main__":
    # Local development only. On Render, Gunicorn starts the app (see README).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
