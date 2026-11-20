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

Files needed: app.py, requirements.txt, logistic.pkl (+ optional categories.json)
Local run : python app.py
Render    : gunicorn app:app
"""
import json
import logging
import math
import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, render_template_string, request

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

app = Flask(__name__, static_folder=None)  # single-file app: no static/templates folders


# ----------------------------------------------------------------------------
# 4. Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML, fields=FIELDS, classes=CLASSES, has_proba=HAS_PROBA)


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
        # Values are put in the exact feature order the model was trained with.
        X = np.array([[row[n] for n in FEATURES]], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hides the harmless "no feature names" warning
            label = str(model.predict(X)[0])
            probabilities = None
            confidence = None
            if HAS_PROBA:
                p = model.predict_proba(X)[0]
                probabilities = {c: round(float(v) * 100, 2) for c, v in zip(CLASSES, p)}
                confidence = round(float(max(p)) * 100, 2)
    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {type(exc).__name__}: {exc}"), 500

    return jsonify(prediction=label, confidence=confidence, probabilities=probabilities)


@app.errorhandler(Exception)
def handle_unexpected(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    log.exception("Unhandled error")
    return jsonify(error=f"Server error: {type(exc).__name__}: {exc}"), 500


# ----------------------------------------------------------------------------
# 5. The whole web page (HTML + CSS + JavaScript) lives here, so this one file
#    is all you need besides requirements.txt and logistic.pkl.
# ----------------------------------------------------------------------------
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#04130e">
  <title>Verdant AI · GenAI Learning Outcome Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
  <style>
{% raw %}
/* ==========================================================================
   Verdant AI – green glassmorphism theme (dark + light)
   ========================================================================== */
:root {
  --g-50:  #ecfdf5;
  --g-200: #a7f3d0;
  --g-300: #6ee7b7;
  --g-400: #34d399;
  --g-500: #10b981;
  --g-600: #059669;
  --g-700: #047857;
  --lime:  #bef264;
  --teal:  #2dd4bf;
  --radius: 24px;
  --ease: cubic-bezier(.2, .8, .2, 1);
}

/* ---------- Dark (default) ---------- */
[data-theme="dark"] {
  --bg1: #03110c; --bg2: #06241a; --bg3: #0a3a2a;
  --text: #e8fff4; --muted: rgba(209, 250, 229, .62);
  --glass: rgba(255, 255, 255, .06);
  --glass-hi: rgba(255, 255, 255, .11);
  --border: rgba(167, 243, 208, .16);
  --input: rgba(2, 20, 14, .55);
  --shadow: 0 24px 60px rgba(0, 0, 0, .45);
  --accent: var(--g-400);
  --accent-text: #052e1c;
  --blob-a: #10b981; --blob-b: #2dd4bf; --blob-c: #84cc16;
  --blob-op: .30;
}

/* ---------- Light ---------- */
[data-theme="light"] {
  --bg1: #f0fdf6; --bg2: #d9f8e6; --bg3: #bdf0d4;
  --text: #06301f; --muted: rgba(6, 48, 31, .62);
  --glass: rgba(255, 255, 255, .55);
  --glass-hi: rgba(255, 255, 255, .8);
  --border: rgba(5, 150, 105, .22);
  --input: rgba(255, 255, 255, .8);
  --shadow: 0 20px 50px rgba(5, 95, 70, .16);
  --accent: var(--g-600);
  --accent-text: #ffffff;
  --blob-a: #34d399; --blob-b: #5eead4; --blob-c: #bef264;
  --blob-op: .55;
}

* { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
[hidden] { display: none !important; }
html { scroll-behavior: smooth; }

body {
  font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text);
  min-height: 100vh; min-height: 100dvh;
  background: linear-gradient(135deg, var(--bg1), var(--bg2), var(--bg3), var(--bg2), var(--bg1));
  background-size: 400% 400%;
  animation: bgFlow 30s ease-in-out infinite;
  overflow-x: hidden;
  transition: color .4s ease;
}
@keyframes bgFlow { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

/* ---------- Floating blobs ---------- */
.bg { position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; }
.blob { position: absolute; border-radius: 50%; filter: blur(70px); opacity: var(--blob-op); animation: drift 22s ease-in-out infinite; will-change: transform; }
.b1 { width: min(46vw, 520px); aspect-ratio: 1; background: var(--blob-a); left: -10%; top: -8%; }
.b2 { width: min(38vw, 440px); aspect-ratio: 1; background: var(--blob-b); right: -8%; top: 28%; animation-delay: -8s; animation-duration: 28s; }
.b3 { width: min(34vw, 380px); aspect-ratio: 1; background: var(--blob-c); left: 30%; bottom: -14%; animation-delay: -14s; animation-duration: 32s; }
@keyframes drift {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  33% { transform: translate3d(60px, -40px, 0) scale(1.1); }
  66% { transform: translate3d(-40px, 50px, 0) scale(.92); }
}

.container {
  position: relative; z-index: 1;
  max-width: 1200px; margin: 0 auto;
  padding: clamp(16px, 3vw, 32px) clamp(14px, 3vw, 28px) max(40px, env(safe-area-inset-bottom));
}

/* ---------- Header ---------- */
.top { display: flex; align-items: center; justify-content: space-between; animation: fadeDown .7s var(--ease) both; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand strong { font-family: "Sora", sans-serif; font-size: 1.05rem; display: block; }
.brand small { color: var(--muted); font-size: .75rem; }
.logo {
  width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; font-size: 1.15rem;
  background: linear-gradient(135deg, var(--g-400), var(--g-600)); color: #fff;
  box-shadow: 0 8px 22px rgba(16, 185, 129, .45);
}
.icon-btn {
  width: 46px; height: 46px; border-radius: 14px; border: 1px solid var(--border); cursor: pointer;
  background: var(--glass); color: var(--text); font-size: 1.05rem;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  transition: transform .3s var(--ease), background .3s ease, box-shadow .3s ease;
}
.icon-btn:hover { transform: translateY(-2px) rotate(12deg); background: var(--glass-hi); box-shadow: 0 8px 22px rgba(16, 185, 129, .35); }
[data-theme="dark"] .light-only, [data-theme="light"] .dark-only { display: none; }

/* ---------- Hero ---------- */
.hero { text-align: center; margin: clamp(22px, 5vw, 48px) auto clamp(22px, 4vw, 40px); max-width: 720px; animation: fadeUp .9s var(--ease) both; }
.pill {
  display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px; font-size: .78rem; font-weight: 500;
  background: var(--glass); border: 1px solid var(--border); color: var(--accent);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}
.hero h1 { font-family: "Sora", sans-serif; font-size: clamp(2rem, 6vw, 3.6rem); line-height: 1.08; letter-spacing: -.02em; margin: 18px 0 14px; }
.grad {
  background: linear-gradient(100deg, var(--g-500), var(--teal), var(--lime), var(--g-400));
  background-size: 250% auto; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shine 7s linear infinite;
}
@keyframes shine { to { background-position: 250% center; } }
.hero p { color: var(--muted); font-size: clamp(.95rem, 2.2vw, 1.08rem); line-height: 1.65; font-weight: 300; }

/* ---------- Layout ---------- */
.layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: clamp(16px, 2.4vw, 26px); align-items: start; }
.side { display: grid; gap: clamp(16px, 2.4vw, 26px); position: sticky; top: 20px; }

/* ---------- Glass card ---------- */
.glass {
  position: relative; border-radius: var(--radius); padding: clamp(18px, 2.6vw, 28px);
  background: linear-gradient(145deg, var(--glass-hi), var(--glass));
  border: 1px solid var(--border);
  backdrop-filter: blur(22px) saturate(150%); -webkit-backdrop-filter: blur(22px) saturate(150%);
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, .18);
  animation: fadeUp .9s var(--ease) both;
}
.side .glass:nth-child(2) { animation-delay: .15s; }
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 20px; }
.card-head h2 { font-family: "Sora", sans-serif; font-size: 1.08rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.card-head h2 i { color: var(--accent); }
.muted { color: var(--muted); font-size: .82rem; }
.center { text-align: center; }

/* ---------- Form ---------- */
.fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px 16px; }
.field label { display: flex; align-items: center; gap: 8px; font-size: .86rem; font-weight: 600; margin-bottom: 8px; }
.field label i { color: var(--accent); width: 16px; text-align: center; font-size: .85rem; }
.control { position: relative; }
.control input, .control select {
  width: 100%; padding: 14px 16px; border-radius: 14px; font: inherit; font-size: .95rem; color: var(--text);
  background: var(--input); border: 1px solid var(--border); outline: none; appearance: none; -webkit-appearance: none;
  transition: border-color .25s ease, box-shadow .25s ease, transform .25s var(--ease), background .25s ease;
}
.control input::placeholder { color: var(--muted); opacity: .7; }
.control input:hover, .control select:hover { border-color: rgba(52, 211, 153, .5); }
.control input:focus, .control select:focus { border-color: var(--g-400); box-shadow: 0 0 0 4px rgba(52, 211, 153, .2); transform: translateY(-1px); }
.control select option { background: #0a2a1f; color: #e8fff4; }
.control select:invalid { color: var(--muted); }
.select-wrap .chev { position: absolute; right: 16px; top: 50%; transform: translateY(-50%); pointer-events: none; color: var(--muted); font-size: .78rem; }
.hint { display: block; margin-top: 7px; color: var(--muted); font-size: .76rem; line-height: 1.4; }
.error-text { display: none; margin-top: 6px; color: #fb7185; font-size: .78rem; font-weight: 500; }
.field.invalid .control input, .field.invalid .control select { border-color: #fb7185; box-shadow: 0 0 0 4px rgba(251, 113, 133, .15); animation: shake .4s; }
.field.invalid .error-text { display: block; }
[data-theme="light"] .error-text { color: #be123c; }
@keyframes shake { 20%, 60% { transform: translateX(-5px); } 40%, 80% { transform: translateX(5px); } }

/* ---------- Buttons ---------- */
.actions { display: flex; gap: 12px; margin-top: 26px; }
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 10px; border: 0; cursor: pointer;
  font: 600 .98rem "Inter", sans-serif; padding: 15px 22px; border-radius: 16px; position: relative; overflow: hidden;
  transition: transform .25s var(--ease), box-shadow .3s ease, background .3s ease;
}
.btn.primary {
  flex: 1; color: var(--accent-text); letter-spacing: .01em;
  background: linear-gradient(110deg, var(--g-400), var(--teal), var(--g-500), var(--lime));
  background-size: 300% 100%; animation: flow 8s linear infinite;
  box-shadow: 0 12px 30px rgba(16, 185, 129, .4);
}
[data-theme="light"] .btn.primary { color: #fff; }
.btn.primary::after { content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%; background: linear-gradient(100deg, transparent, rgba(255, 255, 255, .55), transparent); transform: skewX(-20deg); animation: sweep 4s ease-in-out infinite; }
.btn.primary:hover { transform: translateY(-3px); box-shadow: 0 18px 40px rgba(45, 212, 191, .45); }
.btn:active { transform: scale(.97); }
.btn.ghost { color: var(--text); background: var(--glass); border: 1px solid var(--border); }
.btn.ghost:hover { background: var(--glass-hi); transform: translateY(-2px); box-shadow: 0 10px 24px rgba(16, 185, 129, .2); }
.btn.small { padding: 10px 14px; font-size: .82rem; border-radius: 12px; }
.btn[disabled] { opacity: .65; cursor: progress; }
@keyframes flow { to { background-position: 300% 0; } }
@keyframes sweep { 0% { left: -60%; } 55%, 100% { left: 130%; } }
.link-btn { margin: 14px auto 0; display: flex; align-items: center; gap: 8px; background: none; border: 0; color: var(--muted); cursor: pointer; font: 500 .82rem "Inter", sans-serif; transition: color .2s; }
.link-btn:hover { color: var(--accent); }

/* ---------- Result card ---------- */
.empty { text-align: center; padding: 14px 6px 6px; color: var(--muted); font-size: .9rem; line-height: 1.6; }
.orb {
  width: 84px; height: 84px; margin: 0 auto 16px; border-radius: 50%; display: grid; place-items: center; font-size: 1.9rem; color: #fff;
  background: radial-gradient(circle at 30% 25%, var(--g-300), var(--g-600) 70%);
  box-shadow: 0 0 40px rgba(52, 211, 153, .55); animation: bob 4s ease-in-out infinite;
}
@keyframes bob { 50% { transform: translateY(-8px); box-shadow: 0 0 60px rgba(52, 211, 153, .75); } }

.loader { text-align: center; padding: 18px 0 8px; }
.rings { position: relative; width: 110px; height: 110px; margin: 0 auto 16px; display: grid; place-items: center; color: var(--accent); font-size: 1.7rem; }
.rings span { position: absolute; inset: 0; border-radius: 50%; border: 3px solid transparent; border-top-color: var(--g-400); animation: spin 1.1s linear infinite; }
.rings span:nth-child(2) { inset: 12px; border-top-color: var(--teal); animation-duration: 1.6s; animation-direction: reverse; }
.rings span:nth-child(3) { inset: 24px; border-top-color: var(--lime); animation-duration: 2.1s; }
.rings i { animation: pulse 1.4s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse { 50% { transform: scale(1.2); opacity: .7; } }
.loader p { color: var(--muted); font-weight: 500; }
.dots::after { content: ""; animation: dots 1.4s steps(4) infinite; }
@keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75%, 100% { content: "..."; } }

.result:not([hidden]) { animation: pop .6s cubic-bezier(.3, 1.4, .5, 1) both; }
@keyframes pop { from { opacity: 0; transform: scale(.9) translateY(10px); } to { opacity: 1; transform: none; } }
.donut-wrap { position: relative; width: min(100%, 230px); aspect-ratio: 1; margin: 0 auto 22px; }
.donut { width: 100%; height: 100%; transform: rotate(-90deg); overflow: visible; }
.donut circle { fill: none; stroke-width: 9; stroke-linecap: round; }
.donut .track { stroke: var(--border); }
.donut .value { stroke: var(--class-color, var(--accent)); stroke-dasharray: 326.7; stroke-dashoffset: 326.7; transition: stroke-dashoffset 1.3s var(--ease), stroke .4s; filter: drop-shadow(0 0 8px rgba(52, 211, 153, .7)); }
.donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 2px; }
.label-sm { font-size: .7rem; letter-spacing: .18em; text-transform: uppercase; color: var(--muted); }
.donut-center strong { font-family: "Sora", sans-serif; font-size: clamp(1.6rem, 5vw, 2.1rem); }
.conf { font-size: .82rem; color: var(--muted); }
.conf b { color: var(--accent); font-size: 1rem; }

.probs { display: grid; gap: 14px; }
.prob-top { display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: 6px; }
.prob-top b { font-variant-numeric: tabular-nums; }
.bar { height: 10px; border-radius: 99px; background: var(--border); overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, var(--g-500), var(--teal)); transition: width 1.1s var(--ease); box-shadow: 0 0 12px rgba(52, 211, 153, .6); }
.prob.leading .bar i { background: linear-gradient(90deg, var(--lime), var(--g-400)); }
.prob.leading .prob-top { font-weight: 600; color: var(--accent); }

/* Class colours – all kept in the green family */
.cls-High   { --class-color: #10b981; }
.cls-Medium { --class-color: #a3e635; }
.cls-Low    { --class-color: #2dd4bf; }

/* ---------- History ---------- */
.history { list-style: none; display: grid; gap: 10px; max-height: 300px; overflow-y: auto; padding-right: 4px; }
.history li {
  display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; padding: 12px 14px; border-radius: 14px;
  background: var(--glass); border: 1px solid var(--border); animation: slideIn .5s var(--ease) both;
}
.history .dot { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-size: .8rem; color: #052e1c; background: var(--class-color, var(--g-400)); font-weight: 700; }
.history strong { display: block; font-size: .92rem; }
.history small { color: var(--muted); font-size: .74rem; }
.history .pct { font-weight: 600; font-size: .85rem; color: var(--accent); }
@keyframes slideIn { from { opacity: 0; transform: translateX(18px); } to { opacity: 1; transform: none; } }
.history::-webkit-scrollbar { width: 6px; }
.history::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }

/* ---------- Toasts ---------- */
.toasts { position: fixed; top: max(16px, env(safe-area-inset-top)); right: 16px; z-index: 50; display: grid; gap: 10px; width: min(360px, calc(100vw - 32px)); }
.toast {
  display: flex; align-items: flex-start; gap: 12px; padding: 14px 16px; border-radius: 16px; font-size: .9rem; line-height: 1.45;
  background: var(--glass-hi); border: 1px solid var(--border); color: var(--text);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); box-shadow: var(--shadow);
  animation: toastIn .5s var(--ease) both;
}
[data-theme="dark"] .toast { background: rgba(6, 36, 26, .85); }
.toast i { margin-top: 2px; font-size: 1.05rem; }
.toast.success { border-left: 4px solid var(--g-400); } .toast.success i { color: var(--g-400); }
.toast.error { border-left: 4px solid #fb7185; } .toast.error i { color: #fb7185; }
.toast.info { border-left: 4px solid var(--teal); } .toast.info i { color: var(--teal); }
.toast.out { animation: toastOut .35s ease forwards; }
@keyframes toastIn { from { opacity: 0; transform: translateX(40px); } to { opacity: 1; transform: none; } }
@keyframes toastOut { to { opacity: 0; transform: translateX(40px); } }

@keyframes fadeUp { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: none; } }
@keyframes fadeDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: none; } }

footer { text-align: center; margin-top: 36px; color: var(--muted); font-size: .8rem; }

/* ---------- Responsive ---------- */
@media (max-width: 1000px) {            /* tablet */
  .layout { grid-template-columns: 1fr; }
  .side { position: static; }
}
@media (max-width: 600px) {             /* mobile */
  .fields { grid-template-columns: 1fr; }
  .actions { flex-direction: column; }
  .glass { border-radius: 20px; }
  .brand small { display: none; }
  .blob { filter: blur(55px); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}

{% endraw %}
</style>
  <script>
    // Apply the saved theme before first paint (prevents a flash).
    try { document.documentElement.dataset.theme = localStorage.getItem("theme") || "dark"; } catch (e) {}
  </script>
</head>
<body>
  <div class="bg" aria-hidden="true">
    <span class="blob b1"></span><span class="blob b2"></span><span class="blob b3"></span>
  </div>

  <div class="toasts" id="toasts" aria-live="polite"></div>

  <div class="container">
    <header class="top">
      <div class="brand">
        <span class="logo"><i class="fa-solid fa-leaf"></i></span>
        <div>
          <strong>Verdant AI</strong>
          <small>Machine Learning Studio</small>
        </div>
      </div>
      <button class="icon-btn" id="themeToggle" type="button" aria-label="Toggle dark / light mode">
        <i class="fa-solid fa-moon dark-only"></i>
        <i class="fa-solid fa-sun light-only"></i>
      </button>
    </header>

    <section class="hero">
      <span class="pill"><i class="fa-solid fa-bolt"></i> Logistic Regression · Live inference</span>
      <h1>GenAI Learning<br><span class="grad">Outcome Predictor</span></h1>
      <p>Describe a student's study and AI habits and get an instant prediction with class probabilities.</p>
    </section>

    <main class="layout">
      <!-- ============ FORM ============ -->
      <section class="glass form-card">
        <div class="card-head">
          <h2><i class="fa-solid fa-sliders"></i> Student profile</h2>
          <span class="muted">{{ fields|length }} inputs</span>
        </div>

        <form id="predictForm" novalidate>
          <div class="fields">
            {% for f in fields %}
            <div class="field" data-name="{{ f.name }}">
              <label for="{{ f.name }}"><i class="fa-solid {{ f.icon }}"></i> {{ f.label }}</label>

              {% if f.kind == 'category' %}
              <div class="control select-wrap">
                <select id="{{ f.name }}" name="{{ f.name }}" required>
                  <option value="" selected disabled>Select…</option>
                  {% for opt in f.options %}
                  <option value="{{ opt }}">{{ opt }}</option>
                  {% endfor %}
                </select>
                <i class="fa-solid fa-chevron-down chev"></i>
              </div>
              {% else %}
              <div class="control">
                <input id="{{ f.name }}" name="{{ f.name }}" type="number" inputmode="decimal"
                       min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                       placeholder="Enter value" required>
              </div>
              {% endif %}

              <small class="hint">{{ f.hint }}</small>
              <small class="error-text"></small>
            </div>
            {% endfor %}
          </div>

          <div class="actions">
            <button type="submit" class="btn primary" id="predictBtn">
              <i class="fa-solid fa-wand-magic-sparkles"></i> <span>Predict Outcome</span>
            </button>
            <button type="button" class="btn ghost" id="resetBtn">
              <i class="fa-solid fa-rotate-left"></i> Reset
            </button>
          </div>
        </form>
      </section>

      <!-- ============ RESULT + HISTORY ============ -->
      <aside class="side">
        <section class="glass result-card" id="resultCard">
          <div class="card-head">
            <h2><i class="fa-solid fa-chart-pie"></i> Prediction</h2>
          </div>

          <div class="empty" id="emptyState">
            <div class="orb"><i class="fa-solid fa-seedling"></i></div>
            <p>Fill in the profile and press <b>Predict Outcome</b> to see the result here.</p>
          </div>

          <div class="loader" id="loader" hidden>
            <div class="rings"><span></span><span></span><span></span><i class="fa-solid fa-microchip"></i></div>
            <p>Analysing patterns<span class="dots"></span></p>
          </div>

          <div class="result" id="result" hidden>
            <div class="donut-wrap">
              <svg viewBox="0 0 120 120" class="donut" aria-hidden="true">
                <circle class="track" cx="60" cy="60" r="52"></circle>
                <circle class="value" id="donutValue" cx="60" cy="60" r="52"></circle>
              </svg>
              <div class="donut-center">
                <span class="label-sm">Prediction</span>
                <strong id="predLabel">–</strong>
                {% if has_proba %}<span class="conf"><b id="confValue">0</b>% confident</span>{% endif %}
              </div>
            </div>

            {% if has_proba %}
            <div class="probs" id="probs">
              {% for c in classes %}
              <div class="prob" data-class="{{ c }}">
                <div class="prob-top"><span>{{ c }}</span><b>0%</b></div>
                <div class="bar"><i></i></div>
              </div>
              {% endfor %}
            </div>
            {% endif %}
          </div>
        </section>

        <section class="glass history-card">
          <div class="card-head">
            <div>
              <h2><i class="fa-solid fa-clock-rotate-left"></i> Session history</h2>
              <span class="muted">Predictions made during this session</span>
            </div>
            <button class="btn small ghost" id="csvBtn" type="button">
              <i class="fa-solid fa-download"></i> CSV
            </button>
          </div>
          <ul class="history" id="historyList"></ul>
          <p class="muted center" id="historyEmpty">Your prediction history will appear here.</p>
          <button class="link-btn" id="clearHistory" type="button" hidden>
            <i class="fa-regular fa-trash-can"></i> Clear history
          </button>
        </section>
      </aside>
    </main>

    <footer>
      Built with Flask &amp; scikit-learn · Deployed on Render
    </footer>
  </div>

  <script>
    // Data handed from Flask to the browser script.
    window.APP = {
      fields: {{ fields | tojson }},
      classes: {{ classes | tojson }},
      hasProba: {{ has_proba | tojson }}
    };
  </script>
  <script>
{% raw %}
/* ==========================================================================
   Verdant AI – frontend logic
   window.APP (fields, classes, hasProba) is injected by templates/index.html
   ========================================================================== */
(() => {
  "use strict";

  const { fields, classes, hasProba } = window.APP;
  const $ = (id) => document.getElementById(id);

  const form = $("predictForm");
  const predictBtn = $("predictBtn");
  const emptyState = $("emptyState");
  const loader = $("loader");
  const resultBox = $("result");
  const HISTORY_KEY = "predictionHistory";
  const CIRCUMFERENCE = 2 * Math.PI * 52;   // matches r="52" in the SVG

  /* ------------------------------ Theme ------------------------------ */
  $("themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked */ }
  });

  /* ------------------------------ Toasts ----------------------------- */
  const ICONS = { success: "fa-circle-check", error: "fa-circle-exclamation", info: "fa-circle-info" };
  function toast(type, message, ms = 4200) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<i class="fa-solid ${ICONS[type]}"></i><span></span>`;
    el.querySelector("span").textContent = message;     // textContent = safe
    $("toasts").appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, ms);
  }

  /* --------------------------- Field errors -------------------------- */
  function setFieldError(name, message) {
    const wrap = form.querySelector(`.field[data-name="${name}"]`);
    if (!wrap) return;
    wrap.classList.add("invalid");
    wrap.querySelector(".error-text").textContent = message;
  }
  function clearErrors() {
    form.querySelectorAll(".field.invalid").forEach((f) => f.classList.remove("invalid"));
  }
  form.addEventListener("input", (e) => {
    const wrap = e.target.closest(".field");
    if (wrap) wrap.classList.remove("invalid");
  });

  /* ---------------------------- Collect data ------------------------- */
  function collect() {
    const data = {};
    const missing = [];
    fields.forEach((f) => {
      const value = form.elements[f.name].value.trim();
      data[f.name] = value;
      if (value === "") { missing.push(f.name); setFieldError(f.name, "This field is required."); }
    });
    return { data, missing };
  }

  /* ------------------------------ UI states -------------------------- */
  function show(state) {
    emptyState.hidden = state !== "empty";
    loader.hidden = state !== "loading";
    resultBox.hidden = state !== "result";
  }

  /* ------------------------------ Predict ---------------------------- */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    const { data, missing } = collect();
    if (missing.length) {
      toast("error", "Please fill in all fields before predicting.");
      form.querySelector(".field.invalid input, .field.invalid select")?.focus();
      return;
    }

    predictBtn.disabled = true;
    predictBtn.querySelector("span").textContent = "Predicting…";
    show("loading");
    if (window.innerWidth < 1000) $("resultCard").scrollIntoView({ behavior: "smooth", block: "center" });

    try {
      // Keep the loading animation visible for a moment so it never just flashes.
      const [response] = await Promise.all([
        fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }),
        new Promise((r) => setTimeout(r, 900)),
      ]);

      let payload = {};
      try { payload = await response.json(); } catch (_) { /* non-JSON error page */ }

      if (!response.ok) {
        if (payload.fields) Object.entries(payload.fields).forEach(([k, v]) => setFieldError(k, v));
        throw new Error(payload.error || `Server error (${response.status}).`);
      }

      renderResult(payload);
      addHistory(data, payload);
      toast("success", `Prediction ready: ${payload.prediction}`);
    } catch (err) {
      show(hasLast ? "result" : "empty");
      toast("error", err.message || "Something went wrong. Please try again.");
    } finally {
      predictBtn.disabled = false;
      predictBtn.querySelector("span").textContent = "Predict Outcome";
    }
  });

  /* ------------------------------ Render ----------------------------- */
  let hasLast = false;

  function renderResult({ prediction, confidence, probabilities }) {
    hasLast = true;
    resultBox.className = `result cls-${prediction}`;
    $("predLabel").textContent = prediction;
    show("result");

    // Only show confidence / probability charts if the model really returned them.
    if (hasProba && probabilities) {
      const ring = $("donutValue");
      ring.style.strokeDasharray = CIRCUMFERENCE;
      ring.style.strokeDashoffset = CIRCUMFERENCE;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        ring.style.strokeDashoffset = CIRCUMFERENCE * (1 - confidence / 100);
      }));
      countUp($("confValue"), confidence);

      document.querySelectorAll("#probs .prob").forEach((row) => {
        const cls = row.dataset.class;
        const pct = probabilities[cls] ?? 0;
        row.classList.toggle("leading", cls === prediction);
        row.querySelector("b").textContent = pct.toFixed(1) + "%";
        const bar = row.querySelector(".bar i");
        bar.style.width = "0%";
        requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = pct + "%"; }));
      });
    }
  }

  function countUp(el, target, duration = 1100) {
    const start = performance.now();
    (function tick(now) {
      const p = Math.min(1, (now - start) / duration);
      el.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(1);
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }

  /* ------------------------------ History ---------------------------- */
  function loadHistory() {
    try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || []; } catch (e) { return []; }
  }
  function saveHistory(list) {
    try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch (e) { /* ignore */ }
  }

  function addHistory(inputs, result) {
    const list = loadHistory();
    list.unshift({
      time: new Date().toISOString(),
      prediction: result.prediction,
      confidence: result.confidence,
      inputs,
    });
    saveHistory(list.slice(0, 50));
    renderHistory();
  }

  function renderHistory() {
    const list = loadHistory();
    const ul = $("historyList");
    ul.innerHTML = "";
    list.forEach((item) => {
      const li = document.createElement("li");
      li.className = `cls-${item.prediction}`;
      const when = new Date(item.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      li.innerHTML = `<span class="dot"></span><div><strong></strong><small></small></div><span class="pct"></span>`;
      li.querySelector(".dot").textContent = item.prediction.charAt(0);
      li.querySelector("strong").textContent = item.prediction;
      li.querySelector("small").textContent = when;
      li.querySelector(".pct").textContent = item.confidence != null ? item.confidence.toFixed(1) + "%" : "";
      ul.appendChild(li);
    });
    $("historyEmpty").hidden = list.length > 0;
    $("clearHistory").hidden = list.length === 0;
  }

  $("clearHistory").addEventListener("click", () => {
    saveHistory([]);
    renderHistory();
    toast("info", "History cleared.");
  });

  /* ---------------------------- CSV download ------------------------- */
  const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;

  $("csvBtn").addEventListener("click", () => {
    const list = loadHistory();
    if (!list.length) { toast("info", "No predictions to download yet."); return; }

    const header = ["Time", "Prediction", ...(hasProba ? ["Confidence (%)"] : []), ...fields.map((f) => f.name)];
    const rows = list.slice().reverse().map((item) => [
      item.time,
      item.prediction,
      ...(hasProba ? [item.confidence] : []),
      ...fields.map((f) => item.inputs[f.name]),
    ]);
    const csv = [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n");

    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prediction_history.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast("success", "Prediction history downloaded.");
  });

  /* ------------------------------- Reset ----------------------------- */
  $("resetBtn").addEventListener("click", () => {
    form.reset();
    clearErrors();
    hasLast = false;
    show("empty");
    toast("info", "All fields cleared.");
  });

  /* -------------------------------- Init ----------------------------- */
  show("empty");
  renderHistory();
})();

{% endraw %}
</script>
</body>
</html>
'''


if __name__ == "__main__":
    # Render provides the port in the PORT environment variable.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
