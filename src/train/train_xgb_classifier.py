import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from loguru import logger
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import (GetMorganGenerator, GetMorganFeatureAtomInvGen, GetRDKitFPGenerator,
                                                GetAtomPairGenerator, GetTopologicalTorsionGenerator)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .featurisation import featurise_smiles, sanitise_smiles


def prepare_data(df: pd.DataFrame, generator: rdFingerprintGenerator, seed: int, test_size: float = 0.2) -> tuple:
    """

    :param df:
    :param generator:
    :param seed:
    :param test_size:
    :return:
    """
    logger.info("Preparing data for training")
    # Generate fingerprints
    X = featurise_smiles(smiles=df['SMILES'].to_list(), fp_generator=generator, sanitise=False)
    y = df['LABEL'].to_list()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    return dtrain, dtest, X, y, X_train, y_train, X_test, y_test


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
    results = xgb.cv(dtrain=dtrain, params=params, nfold=5, num_boost_round=1000, early_stopping_rounds=20,
                     metrics="auc", as_pandas=True)
    auc = results['test-auc-mean'].max()
    return {"loss": -1 * auc, 'status': STATUS_OK}


def train_cross_validation_model(dtrain: xgb.DMatrix, seed: int):
    """

    :param dtrain:
    :param seed:
    :return:
    """
    logger.info("Hyperparameter optimisation")

    # k-fold Cross validation
    hyperparams = {"objective": "binary:logistic",
                   "dtrain": dtrain,
                   "alpha": hp.quniform('alpha', 10, 200, 1),
                   "gamma": hp.uniform('gamma', 1, 9),
                   "lambda": hp.uniform('reg_lambda', 0, 1),
                   "colsample_bytree": hp.uniform('colsample_bytree', 0.5, 1),
                   "min_child_weight": hp.quniform('min_child_weight', 0, 10, 1),
                   "max_depth": hp.uniformint("max_depth", 3, 18),
                   "learning_rate": hp.uniform('learning_rate', 0.01, 0.2),
                   "seed": seed,
                   }

    trials = Trials()
    best_hyperparams = fmin(fn=hyperparameterisation,
                            space=hyperparams,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials,
                            rstate=np.random.default_rng(seed)
                            )

    # We need to get the loss for the best parameters.
    # We have to add the training set in again and convert some parameters back into integers
    best_hyperparams["dtrain"] = dtrain
    best_hyperparams["alpha"] = int(best_hyperparams["alpha"])
    best_hyperparams["max_depth"] = int(best_hyperparams["max_depth"])
    best_hyperparams["min_child_weight"] = int(best_hyperparams["min_child_weight"])
    loss = hyperparameterisation(best_hyperparams)

    return best_hyperparams, -1 * loss['loss']


def feature_importance(xgb_clf, X: xgb.DMatrix, out_dir: Path):
    # Get feature importance
    feat_import = xgb_clf.get_booster().get_score(importance_type='weight')
    feat_import = pd.DataFrame.from_dict(feat_import, orient='index', columns=['WEIGHT'])

    # Save feature importance
    feat_file = out_dir / 'featmap.csv'
    feat_import.to_csv(feat_file)

    # Plot feature importance
    plot_file = out_dir / 'importance.png'
    xgb.plot_importance(xgb_clf)
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.savefig(plot_file)

    # SHAP analysis
    explainer = shap.Explainer(xgb_clf)
    shap_file = out_dir / 'shap.csv'
    shap_values_raw = explainer.shap_values(X)
    shap_values = pd.DataFrame(shap_values_raw)
    shap_values.to_csv(shap_file)

    fig = plt.figure()
    shap.plots.force(explainer.expected_value, shap_values_raw[0], X.iloc[0], show=False)
    fig.savefig(shap_file.as_posix().replace('.csv', '.png'), bbox_inches='tight')

    # We need to predict
    # shap_values['PREDICTION'] = xgb_clf.predict(X)
    #
    # shap_final = pd.DataFrame(index=[n for n in range(0, len(shap_values.columns))])
    # for i in [0, 1]:
    #     label = 'ACTIVE' if i == 1 else 'INACTIVE'
    #     temp = shap_values[shap_values['PREDICTION'] == i]
    #     temp = temp.drop(columns=['PREDICTION'])
    #     temp = temp.median().to_frame(name=f'{label}_SHAP_VALUES')
    #     # - Normalize SHAP values
    #     shap_max = temp[f'{label}_SHAP_VALUES'].abs().max()
    #     temp[f'{label}_SHAP_VALUES_NORM'] = temp[f'{label}_SHAP_VALUES'] / shap_max
    #     shap_final = shap_final.merge(temp, left_index=True, right_index=True)
    #
    # # - Format output
    # shap_final.index.name = 'FEATURE'
    # shap_final.to_csv(shap_file)


def train_classifier(df: pd.DataFrame, model_outpath: str, seed: int = 42, save_features: bool = False):
    """

    :param df: A dataframe of 2 columns: SMILES, LABEL
    :param model_outpath:
    :param seed:
    :param save_features:
    :return:
    """
    # Sanitize SMILES
    df = df.rename(columns={'SMILES': 'INPUT_SMILES'})
    df['SMILES'] = df['INPUT_SMILES'].apply(lambda x: sanitise_smiles(x, as_mol=False))
    df = df.dropna(subset=['SMILES'])
    df = df.drop_duplicates(subset=['SMILES'])

    # Set featurisation routines
    fp_size = 2048
    feature_set = {
                   "Morgan2Bits": GetMorganGenerator(radius=2, fpSize=fp_size),
                   "Morgan2FeatBits": GetMorganGenerator(radius=2, fpSize=fp_size,
                                                         atomInvariantsGenerator=GetMorganFeatureAtomInvGen()),
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
        dtrain, _, X, y, X_train, y_train, X_test, y_test = prepare_data(df=df, generator=generator, seed=seed)

        # Cross validate
        best_hyperparams, loss = train_cross_validation_model(dtrain=dtrain, seed=seed)

        # Store best results
        best_results[feature_name] = {"loss": loss, "params": best_hyperparams}

    # Find feature set that has the best loss and use that to create a model
    best_features = max(best_results.keys(), key=lambda x: best_results[x]['loss'])
    auc = best_results[best_features]['loss']
    params = best_results[best_features]['params']
    logger.info(f"Best Features are: {best_features}. AUC: {auc}")
    logger.info(f"Params: {params}")

    # Regenerate training data
    dtrain, _, X, y, X_train, y_train, X_test, y_test = prepare_data(df=df,
                                                                     generator=feature_set[best_features],
                                                                     seed=seed)

    # instantiate the classifier
    logger.info("Validating Model")
    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = xgb_clf.predict(X_test)
    clf_report = classification_report(y_test, y_pred.tolist(), output_dict=True)
    logger.info(f'Model accuracy score: {round(clf_report["accuracy"], 3)}')
    logger.info(f'Classification report: {clf_report["weighted avg"]}')

    # Train model on full dataset
    xgb_clf.fit(X, y)

    # Save full model
    xgb_clf.save_model(f"{model_outpath}.ubj")

    # Show feature importance
    if save_features:
        out_dir = Path(model_outpath).parents[0]
        feature_importance(xgb_clf=xgb_clf, X=X, out_dir=out_dir)

    return xgb_clf

