from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from loguru import logger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator,
    GetMorganFeatureAtomInvGen,
    GetRDKitFPGenerator,
    GetAtomPairGenerator,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml_for_malaria.train.featurization import featurize_smiles, sanitize_smiles


def prepare_data(
        df: pd.DataFrame,
        generator: rdFingerprintGenerator,
        seed: int,
        test_size: float = 0.2,
        sanitise: bool = False,
) -> tuple:
    """
    Prepare data for training. This function will featurize the SMILES strings and split the data into training and
    testing sets. It will also convert the data into DMatrix format for XGBoost.

    :param df: A dataframe containing 2 columns: SMILES, LABEL
    :param generator: RDKit fingerprint generator
    :param seed: Random seed
    :param test_size: Proportion of data to use for testing
    :param sanitise: Boolean flag, if True sanitise the SMILES strings
    :return: Tuple containing DMatrix for training, DMatrix for testing, features, labels, training features,
    """
    logger.info("Preparing data for training")
    # Generate fingerprints
    x = featurize_smiles(
        smiles=df["SMILES"].to_list(), fp_generator=generator, sanitize=sanitise
    )
    y = df["LABEL"].to_list()

    # Prepare train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    # Format train/test split into DMatrix to improve runtime performance
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)

    return dtrain, dtest, x, y, x_train, y_train, x_test, y_test


def hyperparameterisation(params: dict) -> dict:
    """
    Objective function for hyperparameter optimisation. Here we do XGBoost 5-fold cross-validation and return
    -1*AUC as the loss. We convert to negative as the optimisation function minimises the loss.

    :param params: Dictionary for parameters that will be optimised.
    :return: Dictionary containing loss
    """
    # Extract training data from params. We need to remove it from here, so it's not passed in to the params argument
    dtrain = params.pop("dtrain")

    # Perform cross-validation
    results = xgb.cv(
        dtrain=dtrain,
        params=params,
        nfold=5,
        num_boost_round=1000,
        early_stopping_rounds=20,
        metrics="auc",
        as_pandas=True,
    )
    auc = results["test-auc-mean"].max()
    return {"loss": -1 * auc, "status": STATUS_OK}


def train_cross_validation_model(dtrain: xgb.DMatrix, seed: int) -> tuple[dict, float]:
    """
    Perform hyperparameter optimisation using hyperopt. We use the TPE algorithm to find the best hyperparameters
    for our model.

    :param dtrain: Training data in DMatrix format
    :param seed: Random seed
    :return: Tuple containing the best hyperparameters and the loss
    """
    logger.info("Hyperparameter optimisation")

    # k-fold Cross validation
    hyperparams = {
        "objective": "binary:logistic",
        "dtrain": dtrain,
        "alpha": hp.quniform("alpha", 10, 200, 1),
        "gamma": hp.uniform("gamma", 1, 9),
        "lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "max_depth": hp.uniformint("max_depth", 3, 18),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "seed": seed,
    }

    trials = Trials()
    best_hyperparams = fmin(
        fn=hyperparameterisation,
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )

    # We need to get the loss for the best parameters.
    # We have to add the training set in again and convert some parameters back into integers
    best_hyperparams["dtrain"] = dtrain
    best_hyperparams["alpha"] = int(best_hyperparams["alpha"])
    best_hyperparams["max_depth"] = int(best_hyperparams["max_depth"])
    best_hyperparams["min_child_weight"] = int(best_hyperparams["min_child_weight"])
    loss = hyperparameterisation(best_hyperparams)

    return best_hyperparams, -1 * loss["loss"]


def feature_importance(xgb_clf, x: xgb.DMatrix, out_dir: Path):
    """
    Perform feature importance analysis using SHAP values. We will save the feature importance as a CSV file and
    plot the feature importance.

    :param xgb_clf: XGBoost classifier
    :param x: DMatrix containing the features
    :param out_dir: Output directory
    :return:
    """
    # Get feature importance
    feat_import = xgb_clf.get_booster().get_score(importance_type="weight")
    feat_import = pd.DataFrame.from_dict(
        feat_import, orient="index", columns=["WEIGHT"]
    )

    # Save feature importance
    feat_file = out_dir / "featmap.csv"
    feat_import.to_csv(feat_file)

    # Plot feature importance
    plot_file = out_dir / "importance.png"
    xgb.plot_importance(xgb_clf)
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.savefig(plot_file)

    # SHAP analysis
    explainer = shap.Explainer(xgb_clf)
    shap_file = out_dir / "shap.csv"
    shap_values_raw = explainer.shap_values(x)
    shap_values = pd.DataFrame(shap_values_raw)
    shap_values.to_csv(shap_file)

    fig = plt.figure()
    shap.plots.force(
        explainer.expected_value, shap_values_raw[0], x.iloc[0], show=False
    )
    fig.savefig(shap_file.as_posix().replace(".csv", ".png"), bbox_inches="tight")


def train_classifier(
        df: pd.DataFrame, model_outpath: str, seed: int = 42, save_features: bool = False
) -> XGBClassifier:
    """
    Main entry point for training a XGBoost classifier. This function will perform hyperparameter optimisation
    on a set of fingerprints and train a model using the best fingerprint. The model will be saved to disk.

    :param df: A dataframe containing 2 columns: SMILES, LABEL
    :param model_outpath: Output path for the model
    :param seed: Random seed
    :param save_features: Boolean flag, if True save feature importance
    :return: XGBoost classifier
    """
    # Sanitize SMILES
    df = df.rename(columns={"SMILES": "INPUT_SMILES"})
    df["SMILES"] = df["INPUT_SMILES"].apply(lambda x: sanitize_smiles(x, as_mol=False))
    df = df.dropna(subset=["SMILES"])
    df = df.drop_duplicates(subset=["SMILES"])

    # Set featurization routines
    fp_size = 2048
    feature_set = {
        "Morgan2Bits": GetMorganGenerator(radius=2, fpSize=fp_size),
        "Morgan2FeatBits": GetMorganGenerator(
            radius=2,
            fpSize=fp_size,
            atomInvariantsGenerator=GetMorganFeatureAtomInvGen(),
        ),
        "Morgan3Bits": GetMorganGenerator(radius=3, fpSize=fp_size),
        "RDKit": GetRDKitFPGenerator(fpSize=fp_size),
        "AtomPair": GetAtomPairGenerator(fpSize=fp_size),
        # We can't map atoms to this fingerprint so it will not aid in interpretation.
        # "Topological": GetTopologicalTorsionGenerator(fpSize=fp_size),
    }

    # Perform hyperparameter optimisation on all fingerprints to find the best for our model
    best_results = {}
    for feature_name, generator in feature_set.items():
        logger.info(f"Evaluating {feature_name}")

        # Generate Train/Test splits
        dtrain, _, x, y, x_train, y_train, x_test, y_test = prepare_data(
            df=df, generator=generator, seed=seed
        )

        # Cross validate
        best_hyperparams, loss = train_cross_validation_model(dtrain=dtrain, seed=seed)

        # Store best results
        best_results[feature_name] = {"loss": loss, "params": best_hyperparams}

    # Find feature set that has the best loss and use that to create a model
    best_features = max(best_results.keys(), key=lambda i: best_results[i]["loss"])
    auc = best_results[best_features]["loss"]
    params = best_results[best_features]["params"]
    logger.info(f"Best Features are: {best_features}. AUC: {auc}")
    logger.info(f"Params: {params}")

    # Regenerate training data
    dtrain, _, x, y, x_train, y_train, x_test, y_test = prepare_data(
        df=df, generator=feature_set[best_features], seed=seed
    )

    # instantiate the classifier
    logger.info("Validating Model")
    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(x_train, y_train)

    # Make predictions on test data
    y_pred = xgb_clf.predict(x_test)
    clf_report = classification_report(y_test, y_pred.tolist(), output_dict=True)
    logger.info(f'Model accuracy score: {round(clf_report["accuracy"], 3)}')
    logger.info(f'Classification report: {clf_report["weighted avg"]}')

    # Train model on full dataset
    xgb_clf.fit(x, y)

    # Save full model
    xgb_clf.save_model(f"{model_outpath}.ubj")

    # Show feature importance
    if save_features:
        out_dir = Path(model_outpath).parents[0]
        feature_importance(xgb_clf=xgb_clf, x=x, out_dir=out_dir)

    return xgb_clf
