import sys
from ensemble.stacking import Stacking
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
from dataset.src.processed_dataset_dummies import ProcessedDatasetDummies
from optimizer.my_optuna import MyOptuna

from common.my_enum import MLTask
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, roc_auc_score

# ConfigFileName = "config.yaml"
# config_path = Path(__file__).parent.resolve() / "common" / ConfigFileName
config = Utility().load_yaml_config()


def optimize(models):
    optimizer = MyOptuna(config)
    for model in models:
        optimizer.add_model(*model)
    optimizer.optimize_models(MLTask.Classification, True)


def optimize_predict(train_models, predict_model):
    dataset = ProcessedDataset(config["original_dataset"], "titanic")
    dataset.load()

    ensemble = Stacking(config, MLTask.Classification)
    for model in train_models:
        ensemble.add_model(*model)

    result_dataset = ensemble.make_features(forece_make=False)
    result_dataset.set_raw_label(dataset.get_raw_label())
    result_dataset.transform_label()

    optimizer = MyOptuna(config)
    predict_model[1] = result_dataset
    optimizer.add_model(*predict_model)
    optimizer.optimize_models(MLTask.Classification, True)

    ensemble = Stacking(config, MLTask.Classification)
    predict_model[1] = result_dataset
    ensemble.add_model(*predict_model)
    result_dataset = ensemble.make_features(forece_make=False)


def train(models):
    ensemble = Stacking(config, MLTask.Classification)
    for model in models:
        ensemble.add_model(*model)
    result_dataset = ensemble.make_features(forece_make=True)


def predict(train_models, predict_model, evaluation):
    dataset = ProcessedDataset(config["original_dataset"], "titanic")
    dataset.load()

    ensemble = Stacking(
        config,
        MLTask.Classification,
    )
    for model in train_models:
        ensemble.add_model(*model)

    result_dataset = ensemble.make_features(forece_make=False)
    result_dataset.set_raw_label(dataset.get_raw_label())
    result_dataset.transform_label()
    label_encoder = result_dataset.get_label_encoder()

    ensemble.clear_models()
    predict_model[1] = result_dataset
    ensemble.add_model(*predict_model)

    # threshold = config["model"][predict_model[0]]["threshold"]

    final_result = ensemble.forecast()
    pred_data = final_result.get_numpy_data()[0]
    label = final_result.get_numpy_data()[1]
    test_data = final_result.get_numpy_data()[2]

    train_score = evaluation(label, pred_data)
    print(f"Train Result Score: {train_score:.4f}")

    # acc_max = 0.0
    #
    # steps = [x * 0.001 for x in range(250, 750, 1)]
    # y_pred = (pred_data >= 0.5).astype(int)
    # tmp_acc = accuracy_score(label, y_pred)
    # print(f"thre:0.5 {tmp_acc=}")

    # for tmp_threshold in steps:
    #     y_pred = (pred_data >= tmp_threshold).astype(int)
    #     acc = accuracy_score(label, y_pred)
    #     # print(f"Threshold: {tmp_threshold}, Accuracy: {acc}")
    #     if acc >= acc_max:
    #         acc_max = acc
    #         threshold = tmp_threshold
    # print(f"Best threshold: {threshold}, Accuracy: {acc_max}")

    # threshold = 0.5
    # test_data[test_data < threshold] = 0
    # test_data[test_data >= threshold] = 1
    # test_data_bool = (test_data >= threshold).astype(int).reshape(-1)
    # forecast = label_encoder.inverse_transform(test_data_bool.astype(int))

    id = dataset.get_test_id()
    df = pd.DataFrame({"id": id, "activity": test_data.reshape(-1)})
    df.to_csv("submission.tsv", index=False, header=False, sep="\t")

    df2 = pd.DataFrame({"activity": pred_data.reshape(-1)})
    df2.to_csv("submission_pred.tsv", index=False, header=False, sep="\t")
