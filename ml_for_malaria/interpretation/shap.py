import pandas as pd
import shap
import xgboost as xgb

from matplotlib import pyplot as plt


def shap_feature_importance(
    model, features: pd.DataFrame | xgb.DMatrix, out_path: str | None = None
) -> pd.DataFrame:
    """
    Perform SHAP feature importance analysis.

    :param model: XGBoost model
    :param out_path: Output path for SHAP plot
    :return: DataFrame containing the feature importance
    """

    # Get SHAP values
    explainer = shap.Explainer(model)
    shap_values_raw = explainer.shap_values(features)
    shap_values = pd.DataFrame(shap_values_raw)

    # Save SHAP plot
    if out_path:
        fig = plt.figure()
        shap.plots.force(
            explainer.expected_value, shap_values_raw[0], features.iloc[0], show=False
        )
        fig.savefig(out_path, bbox_inches="tight")

    return shap_values
