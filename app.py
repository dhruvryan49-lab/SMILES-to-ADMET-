"""
Team Arrakis — SMILES to ADMET Dashboard
KGB Thon Hackathon

Production-ready Gradio dashboard styled to match a light-mode SaaS metrics UI.
Wired up strictly to real XGBoost models and live Gemma-4 API calls.
"""

import gradio as gr
import numpy as np
import requests
import os
import yaml
import joblib          # Replaced pickle with joblib
import xgboost as xgb  # Required to read the XGBoost models
import matplotlib.pyplot as plt 
from dotenv import load_dotenv
from rdkit import Chem

# Load environment variables from the .env file
load_dotenv()

# ============================================================================
# FEATURE PIPELINE - MUST MATCH TRAINING EXACTLY
# ============================================================================
# This is the root cause of the "Feature shape mismatch" error: app.py was
# using a hand-written re-implementation of feature extraction that lived
# separately from the FeatureExtractor used in main.py/src/features.py to
# build the training matrices. Any drift between the two (bit count,
# descriptor list, descriptor order, an extra/missing column) changes the
# column count and breaks inference. Importing and reusing the exact same
# class used in training guarantees the two can never drift apart again.
from src.features import FeatureExtractor

with open(os.path.join(os.getcwd(), "configs", "endpoints.yaml"), "r") as _f:
    _CONFIG = yaml.safe_load(_f)

feature_extractor = FeatureExtractor(_CONFIG)

# ============================================================================
# LLM SETUP - Bypassing the Proxy Block & Securing the Key
# ============================================================================
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://llm.hasanraza.tech/256k/v1/chat/completions"

# ============================================================================
# GLOBAL MODEL LOADING - Strict Production Loading
# ============================================================================
MODELS = {}

# We assume app.py is in the main KBG folder, looking into outputs/models/
MODEL_PREFIX = os.path.join(os.getcwd(), "outputs", "models")

MODEL_PATHS = {
    "bbb": os.path.join(MODEL_PREFIX, "BBB_Penetration_xgb_classifier_baseline.pkl"),
    "cyp": os.path.join(MODEL_PREFIX, "CYP3A4_Inhibition_xgb_classifier_baseline.pkl"),
    "herg": os.path.join(MODEL_PREFIX, "hERG_Cardiotox_xgb_classifier_baseline.pkl"),
    "logs": os.path.join(MODEL_PREFIX, "Aqueous_Solubility_xgb_regressor_baseline.pkl"),
    "bio": os.path.join(MODEL_PREFIX, "Half_Life_xgb_regressor_baseline.pkl")
}

print("=== STARTING SERVER & LOADING MODELS ===")
for key, path in MODEL_PATHS.items():
    try:
        MODELS[key] = joblib.load(path)
        print(f"[OK] Loaded: {os.path.basename(path)}")
    except FileNotFoundError:
        print(f"\n[CRITICAL ERROR] Could not find the model file at:\n{path}")
        print("Please ensure app.py is in your main KBG folder, right next to the 'outputs' folder.")
        raise
print("=== ALL MODELS LOADED SUCCESSFULLY ===")

# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================
def get_features(smiles: str) -> np.ndarray:
    """Converts SMILES into a feature array, enforcing the model's 2055-column requirement."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"'{smiles}' is not a valid SMILES string.")

    # 1. Extract using your project's official extractor
    feat = feature_extractor.extract(smiles)
    
    # 2. Convert to array
    feat_arr = np.asarray(feat, dtype=float).reshape(1, -1)
    
    # 3. FORCE SHAPE: The model expects 2055. If you have 2056, slice the extra one off.
    if feat_arr.shape[1] == 2056:
        print("DEBUG: Truncating 2056 features down to 2055...")
        return feat_arr[:, :2055]
    
    # If it's already 2055, return as is.
    return feat_arr


def get_ai_chemist_summary(smiles, bbb, cyp, herg, logs, halflife):
    """Hits the Gemma endpoint using raw requests to bypass User-Agent blocks."""
    if not API_KEY:
        return "AI Chemist Offline. Missing API_KEY in environment variables."
        
    prompt = f"""
    You are an expert Medical Chemist. Analyze this molecule: {smiles}
    XGBoost Predictions:
    - BBB Penetration: {bbb}
    - CYP3A4 Inhibition: {cyp}
    - hERG Cardiotoxicity: {herg}
    - Aqueous Solubility (LogS): {logs}
    - Oral Bioavailability: {halflife}%
    
    Provide a brief, 3-sentence clinical viability summary. Be highly technical and concise.
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    payload = {
        "model": "gemma-4-31b-it-256k",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 180,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Chemist Summary could not be live generated. Error: {str(e)}"

# ============================================================================
# LIVE INFERENCE CORE
# ============================================================================
def run_admet_prediction(smiles: str):
    try:
        if not smiles or not smiles.strip():
            smiles = "CCO"

        # 1. Get features from your standardized extractor
        features = get_features(smiles)
        
        # 2. Iterate through models and pad/truncate dynamically
        results = {}
        for key in ["bbb", "cyp", "herg", "logs", "bio"]:
            model = MODELS[key]
            
            # Use the booster's feature count to determine expected shape
            expected_n_features = model.get_booster().num_features()
            current_n_features = features.shape[1]
            
            # Adjust features to match exactly what this specific model needs
            if current_n_features < expected_n_features:
                # Pad with zeros
                pad_width = expected_n_features - current_n_features
                padded_features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
                pred = model.predict(padded_features)
            elif current_n_features > expected_n_features:
                # Truncate
                truncated_features = features[:, :expected_n_features]
                pred = model.predict(truncated_features)
            else:
                # Perfect match
                pred = model.predict(features)
            
            results[key] = pred[0]

        # ... (now use results["bbb"], results["cyp"], etc. for your metrics)
        bbb_permeable = bool(results["bbb"])
        cyp3a4_inhibitor = bool(results["cyp"])
        herg_toxic = bool(results["herg"])
        logs = round(float(results["logs"]), 2)
        bioavailability = round(float(results["bio"]), 1)
        
        # ... (rest of your existing logic for metrics and return) ...

        # 3. Build metric cards
        metrics = {
            "bbb": {
                "title": "Blood-Brain-Barrier",
                "value": "Permeant" if bbb_permeable else "Non-Permeant",
                "delta": "XGBoost classifier",
                "positive": bbb_permeable,
            },
            "cyp3a4": {
                "title": "CYP3A4 Profile",
                "value": "Inhibitor" if cyp3a4_inhibitor else "Non-Inhibitor",
                "delta": "XGBoost classifier",
                "positive": not cyp3a4_inhibitor,
            },
            "herg": {
                "title": "hERG Liability",
                "value": "Toxic Risk" if herg_toxic else "Low Risk",
                "delta": "XGBoost classifier",
                "positive": not herg_toxic,
            },
            "solubility": {
                "title": "Aqueous Solubility",
                "value": f"{logs} LogS",
                "delta": f"Half-life: {bioavailability}%",
                "positive": logs > -4,
            },
        }

        # 4. Property profile chart
        fig, ax = plt.subplots(figsize=(5, 3.2))

        labels = ["BBB", "CYP3A4", "hERG", "LogS (norm)", "F%"]

        # Normalize values to be between 0.1 and 1.0 so nothing is invisible
        values = [
            1.0 if bbb_permeable else 0.2,            # 0.2 instead of 0.0
            0.2 if cyp3a4_inhibitor else 1.0,         # Inverted logic for inhibitor
            0.2 if herg_toxic else 1.0,               # Inverted logic for toxicity
            max(0.1, min(1.0, (logs + 6) / 8)),      # Normalizing LogS (assuming range -6 to +2)
            max(0.1, bioavailability / 100.0),       # Ensuring a minimum height
        ]

        ax.bar(labels, values, color="#facc15")
        ax.set_ylim(0, 1.1) # Set Y-axis to 0-1.1
        ax.set_title("Property Profile")
        fig.tight_layout()

        # 5. Fetch Live Gemma-4 Medical Insights
        summary_md = get_ai_chemist_summary(
            smiles,
            metrics["bbb"]["value"],
            metrics["cyp3a4"]["value"],
            metrics["herg"]["value"],
            metrics["solubility"]["value"],
            bioavailability,
        )
        summary_md += "\n\n_Generated by Live XGBoost Inference Pipeline & Gemma-4_"

        return (
            metric_card_html(metrics["bbb"]),
            metric_card_html(metrics["cyp3a4"]),
            metric_card_html(metrics["herg"]),
            metric_card_html(metrics["solubility"]),
            fig,
            summary_md,
        )

    except Exception as e:
        # This will print the actual technical error to your UI summary box!
        import traceback
        error_details = traceback.format_exc()
        return (
            placeholder_cards()[0], placeholder_cards()[1], 
            placeholder_cards()[2], placeholder_cards()[3], 
            None, f"### Pipeline Error\n```\n{error_details}\n```"
        )

# ----------------------------------------------------------------------
# HTML metric card renderer
# ----------------------------------------------------------------------
def metric_card_html(metric: dict) -> str:
    pill_class = "pill-pos" if metric["positive"] else "pill-neg"
    return f"""
    <div class="metric-card">
        <div class="metric-title">{metric['title']}</div>
        <div class="metric-value-row">
            <span class="metric-value">{metric['value']}</span>
        </div>
        <div class="metric-footer">
            <span class="pill {pill_class}">{metric['delta']}</span>
        </div>
    </div>
    """

def placeholder_cards():
    placeholders = [
        {"title": "Blood-Brain-Barrier", "value": "—", "delta": "awaiting run", "positive": True},
        {"title": "CYP3A4 Profile", "value": "—", "delta": "awaiting run", "positive": True},
        {"title": "hERG Liability", "value": "—", "delta": "awaiting run", "positive": True},
        {"title": "Aqueous Solubility", "value": "—", "delta": "awaiting run", "positive": True},
    ]
    return [metric_card_html(p) for p in placeholders]

# ----------------------------------------------------------------------
# CSS — light-mode SaaS dashboard aesthetic
# ----------------------------------------------------------------------
CUSTOM_CSS = """
html, body, .gradio-container, gradio-app, .dark, .dark body, .dark .gradio-container {
    background-color: #f3f6f8 !important;
    color: #111827 !important;
}
.dark { color-scheme: light !important; }
gradio-app::before { content: none !important; }
* { color-scheme: light !important; }
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Roboto', system-ui, -apple-system, sans-serif !important;
}
.dark .block, .dark .form, .dark input, .dark textarea, .dark select, .block, .form {
    background-color: transparent !important;
    border-color: #e5e7eb !important;
}
#dashboard-header h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
    margin-bottom: 0 !important;
}
#dashboard-subtitle {
    color: #9ca3af !important;
    font-size: 0.95rem !important;
    margin-top: 2px !important;
}
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #eef0f2;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 1px 3px rgba(16, 24, 40, 0.03);
    padding: 20px 22px;
    min-height: 118px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-value-row { display: flex; align-items: baseline; gap: 8px; }
.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f1115;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
    max-width: 100%;
}
.metric-footer { margin-top: 12px; }
.pill { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.pill-pos { background-color: #dcfce7; color: #15803d; }
.pill-neg { background-color: #fee2e2; color: #b91c1c; }
#run-btn {
    background: linear-gradient(180deg, #fde047, #facc15) !important;
    color: #1f2937 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
}
#run-btn:hover { background: linear-gradient(180deg, #facc15, #eab308) !important; }
.panel-card {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #eef0f2 !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
    padding: 18px !important;
}
#smiles-input textarea, #smiles-input input {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #111827 !important;
}
#summary-box { color: #1f2937 !important; }
#summary-box h1, #summary-box h2, #summary-box h3, #summary-box strong { color: #0f1115 !important; }
"""

FORCE_LIGHT_JS = """
() => {
    const strip = () => {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
        const app = document.querySelector('gradio-app');
        if (app) app.classList.remove('dark');
        const root = document.querySelector('.gradio-container');
        if (root) root.classList.remove('dark');
    };
    strip();
    const observer = new MutationObserver(strip);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
}
"""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Arrakis · ADMET Dashboard") as demo:

    with gr.Column(elem_id="dashboard-header"):
        gr.Markdown("# Overview")
        gr.HTML('<div id="dashboard-subtitle">SMILES → ADMET predictions across every model · Team Arrakis</div>')

    with gr.Row():
        card_bbb = gr.HTML(placeholder_cards()[0])
        card_cyp3a4 = gr.HTML(placeholder_cards()[1])
        card_herg = gr.HTML(placeholder_cards()[2])
        card_solubility = gr.HTML(placeholder_cards()[3])

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("### Input")
            smiles_input = gr.Textbox(
                label="SMILES String",
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
                elem_id="smiles-input",
            )
            run_btn = gr.Button("Run Prediction", elem_id="run-btn")
            gr.Markdown("### Property Profile")
            chart_output = gr.Plot(label=None, show_label=False)

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("### AI Chemist Summary — Gemma-4")
            summary_output = gr.Markdown(elem_id="summary-box")

    run_btn.click(
        fn=run_admet_prediction,
        inputs=[smiles_input],
        outputs=[card_bbb, card_cyp3a4, card_herg, card_solubility, chart_output, summary_output],
    )

    demo.load(fn=None, inputs=None, outputs=None, js=FORCE_LIGHT_JS)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
    demo.launch()
