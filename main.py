import my_ml
import sys
from models.mytorch.my_torch import MyTorch
from pathlib import Path
import yaml
from common.utility import Utility

from models.xgboost.my_xgboost import MyXGBoost
from models.lightgbm.my_light_gbm import MyLightGBM
from models.randomforest.my_randomforest import MyRandomForest
from models.knn.my_knn import MyKNearestNeighbors
from models.svc.my_svc import MySupportVectorMachine
from models.logistic_regression.my_logistic_regression import MyLogisticRegression
from dataset.src.processed_dataset import ProcessedDataset
from dataset.src.processed_dataset_xgboost import ProcessedDatasetXGBoost
from sklearn.metrics import root_mean_squared_error

# from dataset.src.processed_dataset_dummies import ProcessedDatasetDummies
from optimizer.my_optuna import MyOptuna

from common.my_enum import MLTask
import pandas as pd
import copy

args = sys.argv
config = Utility().load_yaml_config()
BaseEvaluation = root_mean_squared_error


def make_models():
    dataset = ProcessedDataset(config["original_dataset"], "bento")
    dataset.load()
    dataset_xgboost = ProcessedDatasetXGBoost(
        config["xgboost_dataset"], "xgboost_titani"
    )
    dataset_xgboost.load()
    # dataset_dummies = ProcessedDatasetDummies(
    #     config["dummies_dataset"], "dummies_titanic"
    # )
    # dataset_dummies.load()

    dataset_category = copy.deepcopy(dataset)
    dataset_category.transform_label()
    dataset_category_xgboost = copy.deepcopy(dataset_xgboost)
    dataset_category_xgboost.transform_label()
    my_torch = MyTorch

    mytorch = ["mytorch", dataset, my_torch, BaseEvaluation]
    random_forest = [
        "randomforest",
        dataset_category,
        MyRandomForest,
        BaseEvaluation,
    ]
    light_gbm = [
        "lightgbm",
        dataset_category_xgboost,
        MyLightGBM,
        BaseEvaluation,
    ]
    xgboost = ["xgboost", dataset_category_xgboost, MyXGBoost, BaseEvaluation]
    logistic_regression = [
        "logistic_regression",
        dataset_category,
        MyLogisticRegression,
        BaseEvaluation,
    ]
    knn = ["knn", dataset_category, MyKNearestNeighbors, BaseEvaluation]
    svc = ["svc", dataset_category, MySupportVectorMachine, BaseEvaluation]
    models = {
        "mytorch": mytorch,
        "randomforest": random_forest,
        "lightgbm": light_gbm,
        "xgboost": xgboost,
        "logistic_regression": logistic_regression,
        "knn": knn,
        "svc": svc,
    }

    return models


def make_predict_models():
    logistic_regression = [
        "result_logistic_regression",
        None,
        MyLogisticRegression,
        BaseEvaluation,
    ]
    xgboost = ["result_xgboost", None, MyXGBoost, BaseEvaluation]
    models = {
        "xgboost": xgboost,
        "logistic_regression": logistic_regression,
    }
    return models


def get_train_models(models):
    selected_models = []
    names = [
        # "mytorch",
        # "randomforest",
        # "lightgbm",
        "xgboost",
        # "logistic_regression",
        # "knn",
        # "svc",
    ]
    for name in names:
        selected_models.append(models[name])
    return selected_models


def get_predict_model(models):
    selected_model = models["logistic_regression"]
    # selected_model = models["xgboost"]
    return selected_model


def main():
    train_models = make_models()
    predict_models = make_predict_models()
    train_models = get_train_models(train_models)
    predict_model = get_predict_model(predict_models)

    if args[1] == "optimize":
        my_ml.optimize(train_models)
    elif args[1] == "train":
        my_ml.train(train_models)
    elif args[1] == "optimize_predict":
        my_ml.optimize_predict(train_models, predict_model)
    elif args[1] == "predict":
        my_ml.predict(train_models, predict_model, BaseEvaluation)
    else:
        print(
            "Invalid argument. Use 'optimize', 'optimize_result', 'train', or 'predict'."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
