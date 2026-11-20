"""
GenAI Learning Outcome Predictor - Flask app for logistic.pkl
==============================================================

What I found by inspecting logistic.pkl (without retraining anything):
  * It is a bare scikit-learn LogisticRegression (no preprocessing pipeline inside).
  * It expects 12 numeric features, in the exact order listed in FEATURES below.
  * Output classes: High, Low, Medium  (predict_proba is supported).
  * Three columns are categorical (Major_Category, Primary_Use_Case,
    Institutional_Policy). The model only accepts numbers, so these were encoded
    to integers before training. The encoder itself is NOT stored in the pickle,
    which is why typing text such as 'computer' gave:
        could not convert string to float: 'computer'

The fix: show those columns as dropdowns of category names and convert the chosen
name to its integer code (see CATEGORY SETTINGS below) before predicting.

Local run : python app.py
Render    : gunicorn app:app
"""
import json
import logging
import math
import os
import pickle
import warnings

import pandas as pd
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "categories.json")


# ----------------------------------------------------------------------------
# 1. Load the model (pickle first, joblib as a fallback)
# ----------------------------------------------------------------------------
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


model = load_model(MODEL_PATH)

# Exact feature order the model was trained with (read from the model itself).
FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]
HAS_PROBA = hasattr(model, "predict_proba")


# ----------------------------------------------------------------------------
# 2. CATEGORY SETTINGS  (the only part you may need to edit)
# ----------------------------------------------------------------------------
# For every categorical column, the dropdown shows the names and the model gets
# the matching integer code.
#
# Option A (recommended): create a file named categories.json next to app.py.
#   - A list  -> codes follow sorted (alphabetical) order, exactly like
#                sklearn's LabelEncoder:   {"Primary_Use_Case": ["Coding", "Writing"]}
#   - A dict  -> you choose the codes:      {"Primary_Use_Case": {"Coding": 0, "Writing": 1}}
#   See the notebook snippet in the README / chat for how to generate it
#   from your dataset in one cell.
#
# Option B: edit PLACEHOLDER_CATEGORIES below.
#
# IMPORTANT: the names and codes MUST match what you used while training,
# otherwise predictions will be wrong even though the app runs.
PLACEHOLDER_CATEGORIES = {
    "Major_Category": ["Arts", "Business", "Engineering", "Science"],
    "Primary_Use_Case": ["Coding", "Research", "Studying", "Writing"],
    "Institutional_Policy": ["Allowed", "Banned", "Restricted"],
}

CATEGORICAL_COLUMNS = ["Major_Category", "Primary_Use_Case", "Institutional_Policy"]


def _normalise(options):
    """list -> {name: code} using sorted order (LabelEncoder style); dict -> as given."""
    if isinstance(options, dict):
        return {str(k): int(v) for k, v in options.items()}
    return {str(name): i for i, name in enumerate(sorted(map(str, options)))}


def load_categories():
    source, raw = "placeholder", dict(PLACEHOLDER_CATEGORIES)
    if os.path.exists(CATEGORIES_PATH):
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            raw.update(json.load(f))
        source = "categories.json"
    maps = {col: _normalise(raw[col]) for col in CATEGORICAL_COLUMNS if col in raw}
    return maps, source


CATEGORY_MAPS, CATEGORY_SOURCE = load_categories()
# Paid_Subscription is stored as 0/1 in the model.
CATEGORY_MAPS["Paid_Subscription"] = {"No": 0, "Yes": 1}

if CATEGORY_SOURCE == "placeholder":
    log.warning(
        "Using PLACEHOLDER category lists. Create categories.json with your real "
        "training categories so predictions are correct."
    )


# ----------------------------------------------------------------------------
# 3. Form fields (labels/hints come from your existing UI; bounds are only
#    loose sanity checks so impossible values are rejected - they are NOT
#    taken from the model).
# ----------------------------------------------------------------------------
FIELD_META = {
    "Major_Category":             dict(label="Major Category",            hint="Student's field of study",                         icon="fa-graduation-cap"),
    "Year_of_Study":              dict(label="Year of Study",             hint="Current academic year, e.g. 1, 2, 3, 4",           icon="fa-calendar-check", min=1,  max=10,  step=1,    integer=True),
    "Pre_Semester_GPA":           dict(label="Pre-Semester GPA",          hint="GPA prior to current semester",                    icon="fa-award",          min=0,  max=10,  step=0.01),
    "Weekly_GenAI_Hours":         dict(label="Weekly GenAI Hours",        hint="Hours spent using GenAI per week",                 icon="fa-robot",          min=0,  max=168, step=0.1),
    "Primary_Use_Case":           dict(label="Primary Use Case",          hint="Main GenAI use case",                              icon="fa-bullseye"),
    "Prompt_Engineering_Skill":   dict(label="Prompt Engineering Skill",  hint="Prompting skill score / level",                    icon="fa-wand-magic-sparkles", min=0, max=100, step=0.1),
    "Paid_Subscription":          dict(label="Paid Subscription",         hint="Do you pay for an AI tool subscription?",          icon="fa-crown"),
    "Traditional_Study_Hours":    dict(label="Traditional Study Hours",   hint="Weekly non-AI study hours",                        icon="fa-book-open",      min=0,  max=168, step=0.1),
    "Perceived_AI_Dependency":    dict(label="Perceived AI Dependency",   hint="Self-reported AI dependency level",                icon="fa-link",           min=0,  max=100, step=0.1),
    "Institutional_Policy":       dict(label="Institutional Policy",      hint="Institution's AI policy category",                 icon="fa-building-columns"),
    "Anxiety_Level_During_Exams": dict(label="Anxiety Level During Exams", hint="Exam anxiety score / rating",                    icon="fa-heart-pulse",    min=0,  max=100, step=0.1),
    "Skill_Retention_Score":      dict(label="Skill Retention Score",     hint="Measured or self-rated retention score",           icon="fa-brain",          min=0,  max=100, step=0.1),
}


def build_fields():
    """One field per model feature, in the model's feature order."""
    fields = []
    for name in FEATURES:
        meta = dict(FIELD_META.get(name, {}))
        meta.setdefault("label", name.replace("_", " "))
        meta.setdefault("hint", name)
        meta.setdefault("icon", "fa-circle-dot")
        meta["name"] = name
        if name in CATEGORY_MAPS:
            meta["kind"] = "category"
            meta["options"] = list(CATEGORY_MAPS[name].keys())
        else:
            meta["kind"] = "number"
            meta.setdefault("min", 0)
            meta.setdefault("max", 1_000_000)
            meta.setdefault("step", "any")
        fields.append(meta)
    return fields


FIELDS = build_fields()

app = Flask(__name__)


# ----------------------------------------------------------------------------
# 4. Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", fields=FIELDS, classes=CLASSES, has_proba=HAS_PROBA)


@app.get("/health")
def health():
    return jsonify(status="ok", categories_source=CATEGORY_SOURCE)


def validate(payload):
    """Return (row_dict, errors_dict). Converts every value to what the model needs."""
    row, errors = {}, {}
    for f in FIELDS:
        name, raw = f["name"], payload.get(f["name"])
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue

        if f["kind"] == "category":
            mapping = CATEGORY_MAPS[name]
            if str(raw) not in mapping:
                errors[name] = "Please choose one of the listed options."
            else:
                row[name] = mapping[str(raw)]
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a valid number."
            continue
        if math.isnan(value) or math.isinf(value):
            errors[name] = "Enter a valid number."
        elif not (f["min"] <= value <= f["max"]):
            errors[name] = f"Value must be between {f['min']} and {f['max']}."
        elif f.get("integer") and value != int(value):
            errors[name] = "Enter a whole number."
        else:
            row[name] = value
    return row, errors


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the form values as JSON."), 400

    row, errors = validate(payload)
    if errors:
        return jsonify(error="Please fix the highlighted fields.", fields=errors), 400

    try:
        # A DataFrame keeps the column names/order the model was trained with.
        X = pd.DataFrame([[row[n] for n in FEATURES]], columns=FEATURES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label = str(model.predict(X)[0])
            probabilities = None
            confidence = None
            if HAS_PROBA:
                p = model.predict_proba(X)[0]
                probabilities = {c: round(float(v) * 100, 2) for c, v in zip(CLASSES, p)}
                confidence = round(float(max(p)) * 100, 2)
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="The model could not make a prediction. Please check your inputs."), 500

    return jsonify(prediction=label, confidence=confidence, probabilities=probabilities)


if __name__ == "__main__":
    # Render provides the port in the PORT environment variable.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
