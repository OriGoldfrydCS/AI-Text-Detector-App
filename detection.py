# ============================
# Import Setup
# ============================

import os
import sys
import numpy as np
import joblib
import xgboost as xgb

# Add project root to sys.path to allow relative imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Feature extraction functions
from embeddings import get_bert_embeddings
from features import (
    extract_statistical_features,
    extract_stylistic_features,
    extract_syntactic_features,
    extract_meta_features
)
from perplexity import get_perplexity_scores

# ============================
# Model Loading
# ============================
# Model paths
CHECKPOINT_DIR = r"C:\Users\PC\Documents\pythonprojects\AMLU_fp_app\model"
JOBLIB_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_xgboost.joblib")
JSON_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "xgboost_model.json")

def _patch_xgb_model(m):
    """
    Patches XGBoost model by setting missing attributes to avoid predict_proba() crashes.

    Args:
        m (xgb.XGBClassifier): XGBoost model object.

    Returns:
        xgb.XGBClassifier: Patched model.
    """
    defaults = {
        "use_label_encoder": False,
        "_le": None,
        "gpu_id": 0,
        "predictor": "auto",
        "n_jobs": 1
    }
    for k, v in defaults.items():
        if not hasattr(m, k):
            setattr(m, k, v)
    return m

# Load XGBoost model
if os.path.exists(JOBLIB_MODEL_PATH):
    model = joblib.load(JOBLIB_MODEL_PATH)
    model = _patch_xgb_model(model)
elif os.path.exists(JSON_MODEL_PATH):
    model = xgb.XGBClassifier(use_label_encoder=False, gpu_id=0, predictor="auto")
    model.load_model(JSON_MODEL_PATH)
    model = _patch_xgb_model(model)
else:
    raise FileNotFoundError(f"No XGBoost model found in\n • {JOBLIB_MODEL_PATH}\n • {JSON_MODEL_PATH}")

# ============================
# Main Analysis Function
# ============================

def analyze_text(text: str) -> tuple[float, float, dict]:
    """
    Analyzes the given text and returns human/AI probabilities and supporting metrics.

    The function extracts embeddings and handcrafted features from the input text,
    builds a 1792-dimensional feature vector, and passes it to a trained XGBoost model
    for classification. It also returns key metrics for visualization.

    Args:
        text (str): Input text to analyze.

    Returns:
        tuple:
            - human_prob (float): Probability the text is human-written (0 to 1).
            - ai_prob (float): Probability the text is AI-generated (0 to 1).
            - metrics (dict): Dictionary of 5 key metrics:
                - "Perplexity"
                - "Type-Token Ratio"
                - "Repetition Rate"
                - "Avg Sentence Length"
                - "Avg Word Length"
    """
    # Extract BERT embeddings (shape: 1×768)
    bert_emb = get_bert_embeddings([text])  # shape (1,768)

    # Extract handcrafted features using NLP
    stat2d = extract_statistical_features([text])  # (1,6) - sentence/word length
    style2d = extract_stylistic_features([text])   # (1,6) - repetition
    synt2d = extract_syntactic_features([text])    # (1,7) - parse tree stats
    meta2d = extract_meta_features([text])         # (1,1) - number of lines or chars

    # GPT-2 based perplexity scores (1×1)
    ppl2d = get_perplexity_scores([text]).reshape(1,1)

    # Model-based features (placeholders)
    bs_h = np.zeros((1,1))      # BERTScore (human)
    bs_ai = np.zeros((1,1))     # BERTScore (AI)
    ngram_ol = np.zeros((1,1))  # n-gram overlap
    lik_rat = np.zeros((1,1))   # likelihood ratio

    # Combine all features into one vector (768 + 6 + 6 + 7 + 1 + 1 + 4 = 793 dims)
    X = np.hstack([
        bert_emb,
        stat2d, style2d, synt2d, meta2d, ppl2d,
        bs_h, bs_ai, ngram_ol, lik_rat
    ])  # shape (1,793)

    # Pad to required input shape (XGBoost model trained on 1792 features)
    expected_dims = 1792
    if X.shape[1] < expected_dims:
        padding = np.zeros((1, expected_dims - X.shape[1]))
        X = np.hstack([X, padding])

    # Predict class probabilities: [P(human), P(AI)]
    proba = model.predict_proba(X)[0]
    human_prob, ai_prob = float(proba[0]), float(proba[1])

    # Debug output
    print(f"Perplexity: {float(ppl2d[0,0])}")
    print(f"Type-Token Ratio: {float(style2d[0,0])}")
    print(f"Repetition Rate: {float(style2d[0,2])}")
    print(f"Avg Sentence Length: {float(stat2d[0,0])}")
    print(f"Avg Word Length: {float(stat2d[0,1])}")
    print(f"Avg Parse Depth: {float(synt2d[0,3])}")  # New debug print

    # Sanitize and assemble key metrics
    ttr = float(style2d[0,0])
    if ttr > 100 or ttr < 0: 
        ttr = max(0, min(100, ttr))  
        
    metrics = {
        "Perplexity": float(ppl2d[0,0]),
        "Type-Token Ratio": ttr,
        "Repetition Rate": float(style2d[0,2]),
        "Avg Sentence Length": float(stat2d[0,0]),
        "Avg Word Length": float(stat2d[0,1])
    }

    return human_prob, ai_prob, metrics