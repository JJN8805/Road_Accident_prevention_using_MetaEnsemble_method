'''import torch
import joblib
import numpy as np
from model import FTTransformer  # same class definition


FEATURE_NAMES = joblib.load("feature_names.pkl")
NUM_FEATURES = len(FEATURE_NAMES)


device = "cpu"

# Load models
ft_model = FTTransformer(n_features=NUM_FEATURES)
ft_model.load_state_dict(torch.load("ft_transformer.pth", map_location=device))
ft_model.eval()

rf = joblib.load("random_forest.pkl")
xgb = joblib.load("xgboost.pkl")
meta = joblib.load("meta_classifier.pkl")

def predict_accident(input_array):
    input_array = np.array(input_array).reshape(1, -1)

    with torch.no_grad():
        ft_pred = torch.sigmoid(
            ft_model(torch.tensor(input_array, dtype=torch.float32))
        ).item()

    rf_pred = rf.predict_proba(input_array)[0, 1]
    xgb_pred = xgb.predict_proba(input_array)[0, 1]

    stacked = np.array([[ft_pred, rf_pred, xgb_pred]])
    final = meta.predict(stacked)[0]

    return int(final)'''

import torch
import joblib
import numpy as np
from model import FTTransformer

FEATURE_NAMES = joblib.load("feature_names.pkl")
NUM_FEATURES = len(FEATURE_NAMES)

device = "cpu"

# Load models
ft_model = FTTransformer(n_features=NUM_FEATURES)
ft_model.load_state_dict(
    torch.load("ft_transformer.pth", map_location=device)
)
ft_model.eval()

rf = joblib.load("random_forest.pkl")
xgb = joblib.load("xgboost.pkl")
meta = joblib.load("meta_classifier.pkl")

def predict_batch(X, threshold=0.35):
    X = np.asarray(X, dtype=np.float32)

    with torch.no_grad():
        ft_preds = torch.sigmoid(
            ft_model(torch.tensor(X))
        ).numpy()

    rf_preds = rf.predict_proba(X)[:, 1]
    xgb_preds = xgb.predict_proba(X)[:, 1]

    stacked = np.vstack([ft_preds, rf_preds, xgb_preds]).T
    probs = meta.predict_proba(stacked)[:, 1]

    labels = (probs >= threshold).astype(int)
    return labels, probs


  