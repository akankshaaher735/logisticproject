import io
import os
import pickle
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "logistic.pkl"))

FALLBACK_FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Institutional_Policy",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
]

NUMERIC_HINTS = {
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Prompt_Engineering_Skill",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Skill_Retention_Score",
}

app = Flask(__name__)


def load_trained_model(path: Path) -> Any:
    """Load logistic.pkl using joblib first, falling back to pickle."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    try:
        return joblib.load(path)
    except Exception as joblib_err:
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as pickle_err:
            raise RuntimeError(
                f"Failed to load model from '{path}'. "
                f"Joblib error: {joblib_err}; Pickle error: {pickle_err}"
            ) from pickle_err


@lru_cache(maxsize=1)
def get_model() -> Any:
    return load_trained_model(MODEL_PATH)


def get_expected_features(model: Any) -> list[str]:
    """Extract exact feature names and order from the loaded model."""
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        return [str(col) for col in model.feature_names_in_]
    return list(FALLBACK_FEATURES)


def get_model_classes(model: Any) -> list[str]:
    """Extract target class labels from the loaded model if available."""
    if hasattr(model, "classes_") and model.classes_ is not None:
        return [str(cls) for cls in model.classes_]
    return []


def parse_feature_value(feature_name: str, raw_value: Any) -> Any:
    """Convert raw user input cleanly while preserving numeric/text intent."""
    if raw_value is None:
        raise ValueError(f"Missing value for '{feature_name}'.")

    if isinstance(raw_value, (int, float, np.number)):
        if pd.isna(raw_value):
            raise ValueError(f"Invalid NaN value for '{feature_name}'.")
        return float(raw_value) if isinstance(raw_value, float) else int(raw_value)

    text = str(raw_value).strip()
    if text == "":
        raise ValueError(f"'{feature_name}' cannot be empty.")

    # If the field looks numeric or is known to be numeric, try parsing as number
    if feature_name in NUMERIC_HINTS or _looks_like_number(text):
        try:
            num = float(text)
            if np.isnan(num) or np.isinf(num):
                raise ValueError(f"'{feature_name}' must be a finite number.")
            return int(num) if num.is_integer() and "." not in text else num
        except ValueError:
            if feature_name in NUMERIC_HINTS:
                raise ValueError(
                    f"'{feature_name}' expects a valid numeric value, got '{text}'."
                )

    return text


def _looks_like_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def build_input_dataframe(payload: dict[str, Any], features: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and order all required features into a DataFrame matching the model."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = [feat for feat in features if feat not in payload]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    cleaned_record: dict[str, Any] = {}
    for feat in features:
        cleaned_record[feat] = parse_feature_value(feat, payload.get(feat))

    df = pd.DataFrame([cleaned_record], columns=features)
    return df, cleaned_record


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GenAI Learning & Retention Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-navy: #070b19;
      --bg-indigo: #111936;
      --bg-purple: #221647;
      --card-bg: rgba(17, 25, 54, 0.68);
      --card-border: rgba(167, 139, 250, 0.22);
      --input-bg: rgba(7, 11, 25, 0.62);
      --input-border: rgba(125, 211, 252, 0.22);
      --text-primary: #f5f7ff;
      --text-secondary: #b8c2e8;
      --text-muted: #8895c7;
      --accent-blue: #3b82f6;
      --accent-purple: #8b5cf6;
      --accent-lavender: #c4b5fd;
      --accent-cyan: #22d3ee;
      --success-bg: rgba(34, 211, 238, 0.14);
      --success-border: rgba(34, 211, 238, 0.45);
      --error-bg: rgba(244, 63, 94, 0.15);
      --error-border: rgba(244, 63, 94, 0.5);
      --shadow-glow: 0 20px 50px rgba(8, 13, 34, 0.6), 0 0 36px rgba(139, 92, 246, 0.16);
    }

    [data-theme="light"] {
      --bg-navy: #eef2ff;
      --bg-indigo: #e0e7ff;
      --bg-purple: #ede9fe;
      --card-bg: rgba(255, 255, 255, 0.78);
      --card-border: rgba(109, 40, 217, 0.18);
      --input-bg: rgba(255, 255, 255, 0.85);
      --input-border: rgba(59, 130, 246, 0.25);
      --text-primary: #0f172a;
      --text-secondary: #334155;
      --text-muted: #64748b;
      --accent-blue: #2563eb;
      --accent-purple: #7c3aed;
      --accent-lavender: #6d28d9;
      --accent-cyan: #0891b2;
      --success-bg: rgba(8, 145, 178, 0.12);
      --success-border: rgba(8, 145, 178, 0.35);
      --error-bg: rgba(225, 29, 72, 0.12);
      --error-border: rgba(225, 29, 72, 0.38);
      --shadow-glow: 0 20px 45px rgba(79, 70, 229, 0.12), 0 0 28px rgba(34, 211, 238, 0.12);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text-primary);
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 15%, rgba(139, 92, 246, 0.28), transparent 42%),
        radial-gradient(circle at 88% 18%, rgba(34, 211, 238, 0.22), transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(59, 130, 246, 0.24), transparent 48%),
        linear-gradient(135deg, var(--bg-navy) 0%, var(--bg-indigo) 52%, var(--bg-purple) 100%);
      background-attachment: fixed;
      transition: background 0.35s ease, color 0.25s ease;
      line-height: 1.5;
    }

    .container {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 64px;
    }

    .topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 28px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-icon {
      width: 48px;
      height: 48px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
      display: grid;
      place-items: center;
      color: #ffffff;
      font-weight: 800;
      font-size: 1.15rem;
      box-shadow: 0 10px 25px rgba(139, 92, 246, 0.38);
      animation: floatPulse 4s ease-in-out infinite;
    }

    @keyframes floatPulse {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-3px); }
    }

    h1 {
      font-family: "Plus Jakarta Sans", "Inter", sans-serif;
      font-size: clamp(1.5rem, 2.8vw, 2.2rem);
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, var(--text-primary), var(--accent-lavender), var(--accent-cyan));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 0.94rem;
    }

    .actions-top {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--card-border);
      border-radius: 22px;
      padding: 24px;
      box-shadow: var(--shadow-glow);
      transition: transform 0.25s ease, border-color 0.25s ease, background 0.3s ease;
    }

    .glass-card:hover {
      border-color: rgba(34, 211, 238, 0.38);
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 18px;
    }

    .pill {
      font-size: 0.78rem;
      font-weight: 600;
      padding: 5px 11px;
      border-radius: 999px;
      background: rgba(139, 92, 246, 0.16);
      border: 1px solid rgba(196, 181, 253, 0.28);
      color: var(--accent-lavender);
    }

    .pill.cyan {
      background: rgba(34, 211, 238, 0.14);
      border-color: rgba(34, 211, 238, 0.32);
      color: var(--accent-cyan);
    }

    .grid-main {
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 24px;
      align-items: start;
    }

    @media (max-width: 980px) {
      .grid-main {
        grid-template-columns: 1fr;
      }
    }

    .section-title {
      font-family: "Plus Jakarta Sans", sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .section-desc {
      color: var(--text-secondary);
      font-size: 0.88rem;
      margin-bottom: 20px;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    @media (max-width: 620px) {
      .form-grid {
        grid-template-columns: 1fr;
      }
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .hint {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    input, select {
      width: 100%;
      padding: 11px 13px;
      border-radius: 12px;
      border: 1px solid var(--input-border);
      background: var(--input-bg);
      color: var(--text-primary);
      font-size: 0.92rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
    }

    input:focus, select:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18);
    }

    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 22px;
    }

    button {
      font-family: inherit;
      font-weight: 600;
      font-size: 0.9rem;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.22s ease, opacity 0.2s ease, background 0.25s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    button:hover:not(:disabled) {
      transform: translateY(-2px);
    }

    button:active:not(:disabled) {
      transform: translateY(0px);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .btn-primary {
      flex: 1;
      min-width: 180px;
      padding: 13px 20px;
      color: #ffffff;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
      background-size: 200% 200%;
      box-shadow: 0 10px 24px rgba(99, 102, 241, 0.35);
    }

    .btn-primary:hover:not(:disabled) {
      background-position: 100% 50%;
      box-shadow: 0 14px 30px rgba(34, 211, 238, 0.32);
    }

    .btn-secondary {
      padding: 12px 16px;
      background: rgba(139, 92, 246, 0.16);
      color: var(--text-primary);
      border: 1px solid var(--card-border);
    }

    .btn-ghost {
      padding: 10px 14px;
      background: var(--card-bg);
      color: var(--text-primary);
      border: 1px solid var(--card-border);
    }

    .notification {
      margin-top: 16px;
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 0.88rem;
      font-weight: 500;
      display: none;
      animation: fadeSlideIn 0.28s ease;
    }

    .notification.show {
      display: block;
    }

    .notification.success {
      background: var(--success-bg);
      border: 1px solid var(--success-border);
      color: var(--text-primary);
    }

    .notification.error {
      background: var(--error-bg);
      border: 1px solid var(--error-border);
      color: var(--text-primary);
    }

    @keyframes fadeSlideIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .right-stack {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .loader-box {
      display: none;
      align-items: center;
      gap: 14px;
      padding: 16px;
      border-radius: 16px;
      background: rgba(139, 92, 246, 0.14);
      border: 1px solid rgba(34, 211, 238, 0.3);
      margin-bottom: 14px;
    }

    .loader-box.active {
      display: flex;
      animation: fadeSlideIn 0.25s ease;
    }

    .spinner {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      border: 3px solid rgba(196, 181, 253, 0.25);
      border-top-color: var(--accent-cyan);
      border-right-color: var(--accent-purple);
      animation: spin 0.85s linear infinite;
      flex-shrink: 0;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .result-card {
      animation: fadeSlideIn 0.35s ease;
    }

    .prediction-banner {
      padding: 18px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(139, 92, 246, 0.22), rgba(34, 211, 238, 0.16));
      border: 1px solid rgba(196, 181, 253, 0.35);
      margin-bottom: 18px;
    }

    .prediction-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-cyan);
      font-weight: 700;
    }

    .prediction-value {
      font-family: "Plus Jakarta Sans", sans-serif;
      font-size: 1.85rem;
      font-weight: 800;
      margin: 4px 0;
    }

    .confidence-text {
      font-size: 0.9rem;
      color: var(--text-secondary);
    }

    .prob-item {
      margin-bottom: 12px;
    }

    .prob-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 5px;
    }

    .progress-track {
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.18);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
      transition: width 0.65s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .history-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }

    .history-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 360px;
      overflow-y: auto;
    }

    .history-item {
      padding: 12px 14px;
      border-radius: 14px;
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      font-size: 0.85rem;
      animation: fadeSlideIn 0.25s ease;
    }

    .history-meta {
      color: var(--text-muted);
      font-size: 0.76rem;
    }

    .empty-state {
      color: var(--text-muted);
      font-size: 0.88rem;
      text-align: center;
      padding: 20px 10px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="topbar">
      <div class="brand">
        <div class="brand-icon">AI</div>
        <div>
          <h1>GenAI Study & Retention Intelligence</h1>
          <p class="subtitle">Real-time inference powered by your trained <code>logistic.pkl</code> model</p>
        </div>
      </div>
      <div class="actions-top">
        <button id="themeToggle" class="btn-ghost" type="button" aria-label="Toggle dark or light theme">
          <span id="themeIcon">☀️</span>
          <span id="themeLabel">Light Mode</span>
        </button>
      </div>
    </header>

    <div class="glass-card" style="margin-bottom: 24px;">
      <div class="badge-row">
        <span class="pill">Model: LogisticRegression (logistic.pkl)</span>
        <span class="pill cyan">12 Exact Features</span>
        <span class="pill" id="classesBadge">Classes: Loading...</span>
      </div>
      <p class="subtitle">
        Enter student academic, GenAI usage, exam anxiety, and retention attributes below. All inputs are validated and sent in the exact feature order expected by your model.
      </p>
    </div>

    <div class="grid-main">
      <section class="glass-card">
        <h2 class="section-title">Interactive Prediction Form</h2>
        <p class="section-desc">Fill in each feature accurately and click <strong>Predict Outcome</strong>.</p>

        <form id="predictionForm" novalidate>
          <div class="form-grid" id="dynamicFields"></div>

          <div class="btn-row">
            <button type="submit" id="predictBtn" class="btn-primary">
              <span>⚡ Predict Outcome</span>
            </button>
            <button type="button" id="resetBtn" class="btn-secondary">
              ↺ Reset Fields
            </button>
          </div>
        </form>

        <div id="notificationBox" class="notification" role="alert"></div>
      </section>

      <div class="right-stack">
        <section class="glass-card">
          <h2 class="section-title">Prediction Output</h2>
          <p class="section-desc">Live inference result and class probability breakdown.</p>

          <div id="loaderBox" class="loader-box" aria-live="polite">
            <div class="spinner"></div>
            <div>
              <div style="font-weight: 700; font-size: 0.9rem;">Running Model Inference...</div>
              <div class="hint">Validating features and computing class probabilities</div>
            </div>
          </div>

          <div id="resultContainer" class="result-card">
            <div class="empty-state">No prediction generated yet. Submit the form to see results.</div>
          </div>
        </section>

        <section class="glass-card">
          <div class="history-header">
            <div>
              <h2 class="section-title" style="margin-bottom: 2px;">Session History</h2>
              <p class="hint">Predictions made during this session</p>
            </div>
            <button type="button" id="downloadCsvBtn" class="btn-secondary" disabled>
              ⬇ Download CSV
            </button>
          </div>

          <div id="historyList" class="history-list">
            <div class="empty-state">Your prediction history will appear here.</div>
          </div>
        </section>
      </div>
    </div>
  </div>

  <script>
    const FEATURE_CONFIG = [
      { name: "Major_Category", label: "Major Category", type: "text", placeholder: "e.g., Computer Science", hint: "Academic major category" },
      { name: "Year_of_Study", label: "Year of Study", type: "number", step: "1", placeholder: "e.g., 2", hint: "Current year of study" },
      { name: "Pre_Semester_GPA", label: "Pre-Semester GPA", type: "number", step: "0.01", placeholder: "e.g., 3.45", hint: "GPA prior to current semester" },
      { name: "Weekly_GenAI_Hours", label: "Weekly GenAI Hours", type: "number", step: "0.1", placeholder: "e.g., 6.5", hint: "Hours spent using GenAI per week" },
      { name: "Primary_Use_Case", label: "Primary Use Case", type: "text", placeholder: "e.g., Coding & Debugging", hint: "Main GenAI use case" },
      { name: "Prompt_Engineering_Skill", label: "Prompt Engineering Skill", type: "number", step: "0.1", placeholder: "e.g., 4", hint: "Prompting skill score/level" },
      { name: "Paid_Subscription", label: "Paid Subscription", type: "text", placeholder: "e.g., Yes or No (or 1/0)", hint: "Paid AI tool subscription status" },
      { name: "Traditional_Study_Hours", label: "Traditional Study Hours", type: "number", step: "0.1", placeholder: "e.g., 10.0", hint: "Weekly non-AI study hours" },
      { name: "Perceived_AI_Dependency", label: "Perceived AI Dependency", type: "number", step: "0.1", placeholder: "e.g., 3", hint: "Self-reported AI dependency level" },
      { name: "Institutional_Policy", label: "Institutional Policy", type: "text", placeholder: "e.g., Allowed with Citation", hint: "Institution AI policy category" },
      { name: "Anxiety_Level_During_Exams", label: "Anxiety Level During Exams", type: "number", step: "0.1", placeholder: "e.g., 3", hint: "Exam anxiety score/rating" },
      { name: "Skill_Retention_Score", label: "Skill Retention Score", type: "number", step: "0.1", placeholder: "e.g., 78.5", hint: "Measured or self-rated retention score" }
    ];

    const fieldsContainer = document.getElementById("dynamicFields");
    const form = document.getElementById("predictionForm");
    const resetBtn = document.getElementById("resetBtn");
    const predictBtn = document.getElementById("predictBtn");
    const loaderBox = document.getElementById("loaderBox");
    const resultContainer = document.getElementById("resultContainer");
    const notificationBox = document.getElementById("notificationBox");
    const historyList = document.getElementById("historyList");
    const downloadCsvBtn = document.getElementById("downloadCsvBtn");
    const themeToggle = document.getElementById("themeToggle");
    const themeIcon = document.getElementById("themeIcon");
    const themeLabel = document.getElementById("themeLabel");
    const classesBadge = document.getElementById("classesBadge");

    let predictionHistory = [];

    function renderFormFields() {
      fieldsContainer.innerHTML = FEATURE_CONFIG.map((cfg) => `
        <div class="field">
          <label for="${cfg.name}">${cfg.label}</label>
          <input
            id="${cfg.name}"
            name="${cfg.name}"
            type="${cfg.type}"
            ${cfg.step ? `step="${cfg.step}"` : ""}
            placeholder="${cfg.placeholder}"
            required
          />
          <span class="hint">${cfg.hint} (${cfg.name})</span>
        </div>
      `).join("");
    }

    function showNotification(message, type = "success") {
      notificationBox.textContent = message;
      notificationBox.className = `notification show ${type}`;
    }

    function hideNotification() {
      notificationBox.className = "notification";
      notificationBox.textContent = "";
    }

    function renderResult(data) {
      const hasProba = Boolean(data.has_probabilities) && Array.isArray(data.probabilities) && data.probabilities.length > 0;
      const confidenceHtml = hasProba && typeof data.confidence === "number"
        ? `<div class="confidence-text">Confidence Score: <strong>${(data.confidence * 100).toFixed(2)}%</strong></div>`
        : `<div class="confidence-text">Confidence score not shown (probability output unavailable for this input/model).</div>`;

      const barsHtml = hasProba
        ? `<div style="margin-top: 14px;">
            <div style="font-size: 0.84rem; font-weight: 700; margin-bottom: 10px;">Class Probability Breakdown</div>
            ${data.probabilities.map((item) => {
              const pct = (Number(item.probability) * 100).toFixed(2);
              return `
                <div class="prob-item">
                  <div class="prob-header">
                    <span>${escapeHtml(String(item.class))}</span>
                    <span>${pct}%</span>
                  </div>
                  <div class="progress-track">
                    <div class="progress-fill" data-width="${pct}"></div>
                  </div>
                </div>
              `;
            }).join("")}
          </div>`
        : "";

      resultContainer.innerHTML = `
        <div class="prediction-banner">
          <div class="prediction-label">Predicted Class</div>
          <div class="prediction-value">${escapeHtml(String(data.prediction))}</div>
          ${confidenceHtml}
        </div>
        ${barsHtml}
      `;

      requestAnimationFrame(() => {
        document.querySelectorAll(".progress-fill").forEach((el) => {
          el.style.width = `${el.dataset.width}%`;
        });
      });
    }

    function renderHistory() {
      if (predictionHistory.length === 0) {
        historyList.innerHTML = `<div class="empty-state">Your prediction history will appear here.</div>`;
        downloadCsvBtn.disabled = true;
        return;
      }

      downloadCsvBtn.disabled = false;
      historyList.innerHTML = predictionHistory.map((entry, idx) => {
        const confStr = typeof entry.confidence === "number"
          ? ` • ${(entry.confidence * 100).toFixed(1)}% conf.`
          : "";
        return `
          <div class="history-item">
            <div>
              <div><strong>#${predictionHistory.length - idx} → ${escapeHtml(String(entry.prediction))}</strong>${confStr}</div>
              <div class="history-meta">${escapeHtml(entry.timestamp)}</div>
            </div>
          </div>
        `;
      }).join("");
    }

    function escapeHtml(str) {
      return str
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function loadModelMetadata() {
      try {
        const res = await fetch("/api/model-info");
        if (!res.ok) return;
        const info = await res.json();
        if (Array.isArray(info.classes) && info.classes.length > 0) {
          classesBadge.textContent = `Classes: ${info.classes.join(", ")}`;
        }
      } catch (_) {
        // Ignore non-critical metadata fetch error
      }
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideNotification();

      const payload = {};
      for (const cfg of FEATURE_CONFIG) {
        const inputEl = document.getElementById(cfg.name);
        payload[cfg.name] = inputEl.value;
      }

      predictBtn.disabled = true;
      loaderBox.classList.add("active");

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.error || "Prediction failed due to an unexpected error.");
        }

        renderResult(data);
        showNotification(`Prediction completed: ${data.prediction}`, "success");

        predictionHistory.unshift({
          timestamp: new Date().toLocaleTimeString(),
          prediction: data.prediction,
          confidence: data.confidence,
          inputs: data.inputs
        });
        renderHistory();
      } catch (err) {
        showNotification(err.message || "Error communicating with server.", "error");
      } finally {
        loaderBox.classList.remove("active");
        predictBtn.disabled = false;
      }
    });

    resetBtn.addEventListener("click", () => {
      form.reset();
      hideNotification();
      showNotification("All input fields have been reset.", "success");
    });

    downloadCsvBtn.addEventListener("click", async () => {
      if (predictionHistory.length === 0) return;
      try {
        const response = await fetch("/download-csv", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ history: predictionHistory })
        });
        if (!response.ok) throw new Error("Failed to export CSV.");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "prediction_history.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        showNotification("Downloaded prediction_history.csv successfully.", "success");
      } catch (err) {
        showNotification(err.message || "CSV download failed.", "error");
      }
    });

    themeToggle.addEventListener("click", () => {
      const htmlEl = document.documentElement;
      const nextTheme = htmlEl.getAttribute("data-theme") === "dark" ? "light" : "dark";
      htmlEl.setAttribute("data-theme", nextTheme);
      if (nextTheme === "light") {
        themeIcon.textContent = "🌙";
        themeLabel.textContent = "Dark Mode";
      } else {
        themeIcon.textContent = "☀️";
        themeLabel.textContent = "Light Mode";
      }
    });

    renderFormFields();
    renderHistory();
    loadModelMetadata();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/model-info", methods=["GET"])
def model_info():
    try:
        model = get_model()
        return jsonify({
            "success": True,
            "model_type": type(model).__name__,
            "features": get_expected_features(model),
            "classes": get_model_classes(model),
            "supports_predict_proba": callable(getattr(model, "predict_proba", None)),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = get_model()
        expected_features = get_expected_features(model)
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"success": False, "error": "Invalid or missing JSON payload."}), 400

        input_df, cleaned_inputs = build_input_dataframe(payload, expected_features)

        pred_array = model.predict(input_df)
        prediction_value = pred_array[0]
        if isinstance(prediction_value, np.generic):
            prediction_value = prediction_value.item()

        has_probabilities = False
        confidence_score = None
        probability_breakdown: list[dict[str, Any]] = []

        if callable(getattr(model, "predict_proba", None)):
            try:
                proba_matrix = model.predict_proba(input_df)
                if proba_matrix is not None and len(proba_matrix) > 0:
                    probs = proba_matrix[0]
                    classes = get_model_classes(model)
                    if not classes:
                        classes = [f"Class_{i}" for i in range(len(probs))]

                    for cls_label, prob_val in zip(classes, probs):
                        probability_breakdown.append({
                            "class": cls_label,
                            "probability": float(prob_val),
                        })

                    confidence_score = float(np.max(probs))
                    has_probabilities = True
            except Exception:
                # Do not fabricate confidence scores if predict_proba fails
                has_probabilities = False
                confidence_score = None
                probability_breakdown = []

        return jsonify({
            "success": True,
            "prediction": str(prediction_value),
            "has_probabilities": has_probabilities,
            "confidence": confidence_score,
            "probabilities": probability_breakdown,
            "inputs": cleaned_inputs,
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Inference error: {exc}"}), 500


@app.route("/download-csv", methods=["POST"])
def download_csv():
    try:
        data = request.get_json(silent=True) or {}
        history = data.get("history", [])
        if not isinstance(history, list) or len(history) == 0:
            return jsonify({"success": False, "error": "No prediction history to export."}), 400

        rows = []
        for item in history:
            row = {
                "timestamp": item.get("timestamp", ""),
                "prediction": item.get("prediction", ""),
                "confidence": item.get("confidence", ""),
            }
            inputs = item.get("inputs", {})
            if isinstance(inputs, dict):
                row.update(inputs)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        return app.response_class(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to generate CSV: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
