import argparse
import logging

import mlflow
import numpy as np
from catboost import CatBoostClassifier
from mlflow.models.signature import infer_signature
from feature_selection import FeatureSelector
from sklearn.metrics import roc_auc_score
import pickle
from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig


class ModelTrainer:
    EXPERIMENT_NAME = "catboost"

    @staticmethod
    def train_model(prob_config: ProblemConfig, model_params, add_captured_data=False):
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelTrainer.EXPERIMENT_NAME}"
        )
        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        logging.info(f"loaded {len(train_x)} samples")

        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            logging.info(f"added {len(captured_x)} captured samples")

        # Do feature selection
        all_cols = prob_config.numerical_cols + prob_config.categorical_cols
        quasi_constant_feat = FeatureSelector.filter_quasi_constant(train_x, prob_config.numerical_cols, prob_config.categorical_cols)
        logging.info(f"Removed quasi-constant: {quasi_constant_feat}")
        remained_cols = [col for col in all_cols if col not in quasi_constant_feat]

        remained_cols = FeatureSelector.feature_importance_random_forest(train_x[remained_cols], train_y)
        logging.info(f"Keep {len(remained_cols)} for training")
        # save selected features
        with open(prob_config.selected_features_path, "wb") as f:
            pickle.dump(remained_cols, f)

        model = CatBoostClassifier(**model_params)
        model.fit(train_x[remained_cols], train_y)
        logging.info(f"Keep {len(remained_cols)} for training")
        # evaluate
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        predictions = model.predict(test_x[remained_cols])
        auc_score = roc_auc_score(test_y, predictions)
        metrics = {"test_auc": auc_score}
        logging.info(f"metrics: {metrics}")

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x[remained_cols], predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        mlflow.end_run()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    ModelTrainer.train_model(
        prob_config, model_config, add_captured_data=args.add_captured_data
    )
