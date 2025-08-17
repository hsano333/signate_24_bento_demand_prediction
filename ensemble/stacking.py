from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import shutil

from pathlib import Path
from common.utility import Utility

from dataset.src.combined_dataset import CombinedDataset
from dataset.src.simple_dataset import SimpleDataset
from common.my_enum import MLTask
import torch
import numpy as np
import os
import yaml


class Stacking:
    RANDAOM_STATE = 42
    DefaultDirectory = "making_features"

    def __init__(self, config):
        self.config = config
        self.dataset_models = {}
        self.dir_path = Path(__file__).parent.resolve() / self.DefaultDirectory
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.KFoldSplit = self.config["stacking"]["k_fold"]
        self.RANDAOM_STATE = self.config["stacking"]["random_state"]
        # self.model_config = self.config["stacking"]["model_config"]
        # self.ForceMake = self.config["stacking"]["force_make"]
        print(f"{self.KFoldSplit=}, {self.RANDAOM_STATE=}")

    def delete_all_files(self, path):
        print(f"delete_all_files: {path=}")
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  # ファイルやシンボリックリンクを削除
            elif item.is_dir():
                shutil.rmtree(item)

    def add_model(self, name, dataset, model):
        self.dataset_models[name] = (dataset, model)

    def clear_models(self):
        self.dataset_models.clear()

    def forecast(self):
        if len(self.dataset_models) != 1:
            raise ValueError(
                "Stacking requires exactly one dataset and one model to learn."
            )

        name = list(self.dataset_models.keys())[0]
        (dataset, model) = self.dataset_models[name]
        model_name = name
        dataset_name = dataset.get_name()

        name = dataset_name + "_" + model_name + "_final" + str(self.KFoldSplit)
        directory = self.dir_path / name
        file_cv_trained_path = (directory / "cv_trained.npz").as_posix()
        file_cv_new_test_path = (directory / "cv_new_test.npz").as_posix()
        if directory.exists() is False:
            directory.mkdir(parents=True, exist_ok=True)

        directory.mkdir(parents=True, exist_ok=True)
        (pred, test) = self.learn_final(model, model_name, dataset, directory)
        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
        np.savez(file_cv_new_test_path, pred=pred, test=test)

        combined_dataset = CombinedDataset(
            pred.numpy(), dataset, test.numpy(), "stacking_combined"
        )
        return combined_dataset

    def make_directory(self, dataset_name, model_name):
        name = dataset_name + "_" + model_name + "_k" + str(self.KFoldSplit)
        directory = self.dir_path / name
        file_cv_trained_path = (directory / "cv_trained.npz").as_posix()
        file_cv_new_test_path = (directory / "cv_new_test.npz").as_posix()
        Utility().make_kfold_directory(directory, self.KFoldSplit)
        # if directory.exists():
        #     dir_count = len([f for f in directory.iterdir() if f.is_dir()])
        #     if dir_count != self.KFoldSplit:
        #         self.delete_all_files(directory)
        #         print("delete file in " + directory.as_posix())
        # else:
        #     directory.mkdir(parents=True, exist_ok=True)
        return (directory, file_cv_trained_path, file_cv_new_test_path)

    def make_param(self, task=MLTask.Classification, forece_make=False):
        preds = []
        tests = []

        for name, (dataset, model) in self.dataset_models.items():
            model_name = name
            dataset_name = dataset.get_name()

            (directory, file_cv_trained_path, file_cv_new_test_path) = (
                self.make_directory(dataset_name, model_name)
            )

            if not os.path.isfile(file_cv_trained_path) or forece_make:
                # directory.mkdir(parents=True, exist_ok=True)
                (pred, test) = self.make_new_feature(
                    model, model_name, dataset, task, directory
                )
                np.savez(file_cv_trained_path, pred=pred, test=test)
                print("make:" + file_cv_trained_path)
            else:
                data = np.load(file_cv_trained_path)
                pred = data["pred"]
                test = data["test"]
                print("load:" + file_cv_trained_path)

            preds.append(pred)
            tests.append(test)

        result_pred = np.hstack(preds)
        result_test = np.hstack(tests)
        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
        np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)

        combined_dataset = CombinedDataset(
            result_pred, dataset, result_test, "stacking_combined"
        )
        return combined_dataset

    def make_features(self, task=MLTask.Classification, forece_make=False):
        preds = []
        tests = []

        for name, (dataset, model) in self.dataset_models.items():
            model_name = name
            dataset_name = dataset.get_name()
            (directory, file_cv_trained_path, file_cv_new_test_path) = (
                self.make_directory(dataset_name, model_name)
            )

            if not os.path.isfile(file_cv_trained_path) or forece_make:
                directory.mkdir(parents=True, exist_ok=True)
                (pred, test) = self.make_new_feature(
                    model, model_name, dataset, task, directory
                )
                np.savez(file_cv_trained_path, pred=pred, test=test)
                print("make:" + file_cv_trained_path)
            else:
                data = np.load(file_cv_trained_path)
                pred = data["pred"]
                test = data["test"]
                print("load:" + file_cv_trained_path)

            # print(f"{pred.shape=}, {test.shape=}, {len(preds)=}, {len(tests)=}")
            # print(f"{pred[0:10]=}, {test[0:10]=}")
            preds.append(pred.reshape(-1, 1))
            tests.append(test.reshape(-1, 1))

        result_pred = np.hstack(preds)
        result_test = np.hstack(tests)
        # ラベルはすべてのデータセットで同じなので、最初のデータセットから取得
        (_, label, _) = next(iter(self.dataset_models.values()))[0].get_numpy_data()
        np.savez(file_cv_new_test_path, pred=result_pred, test=result_test)

        combined_dataset = CombinedDataset(
            result_pred, dataset, result_test, "stacking_combined"
        )
        return combined_dataset

    def learn_final(self, model, model_name, dataset, directory):
        # kf = self.get_kf(task)
        (data, label, test) = dataset.get_numpy_data()
        label_number = dataset.get_label_number()

        tmp_name = dataset.get_name()
        preds = []
        preds_test = []
        va_idxes = []
        label_num = dataset.get_label_number()

        if label.ndim == 1:
            split_label = label
        else:
            split_label = label[:, 0]

        tr_idx = 0
        val_idx = 0
        i = 0
        # for i, (tr_idx, val_idx) in enumerate(kf.split(data, split_label)):
        k_directory = directory / f"fold_{i + 1}"
        va_idxes.append(val_idx)
        # train_data, val_data = data[tr_idx], data[val_idx]
        # train_label, val_label = label[tr_idx], label[val_idx]
        train_data, val_data = data, data
        train_label, val_label = label, label
        new_train = SimpleDataset(
            train_data,
            train_label,
            label_number,
            tmp_name + "_train" + str(i + 1),
        )
        new_val = SimpleDataset(
            val_data,
            val_label,
            label_number,
            tmp_name + "_val" + str(i + 1),
        )
        new_test = SimpleDataset(
            test,
            None,
            label_number,
            tmp_name + "_test" + str(i + 1),
        )

        model_config = self.config["model"][model_name]
        model_config["label_number"] = label_num
        model_config["is_optimize"] = False
        new_model = model(
            model_config, k_directory.as_posix(), new_train, new_val, new_test
        )
        params = model_config.get("learned_params", {})
        new_model.learn(0, params)
        new_model.reload()

        pred = new_model.forecast(torch.tensor(val_data.astype(np.float32)))
        preds.append(pred.numpy())

        pred_test = new_model.forecast(torch.tensor(test.astype(np.float32)))
        preds_test.append(pred_test.numpy())
        # print(f"{preds_test[-1].shape=}, {len(preds_test)=}, {preds_test[0:10]=}")
        #######################################################
        return (pred, pred_test)

        # va_idxes = np.concatenate(va_idxes)
        # preds = np.concatenate(preds, axis=0)
        # order = np.argsort(va_idxes)
        # pred_train = preds[order]
        # print(
        #     f"{preds.shape=}, {va_idxes.shape=}, {len(preds)=}, {pred_train.shape=}, {preds_test[-1].shape=}, {len(preds_test)=}"
        # )
        # print(f"{pred_train[0:10]=}")
        # print(f"{preds_test[0][0:10]=}")
        # print(f"{preds_test[1][0:10]=}")
        # print(f"{preds_test[2][0:10]=}")
        # print(f"{preds_test[3][0:10]=}")
        # print(f"{preds_test[4][0:10]=}")
        # preds_test = np.mean(preds_test, axis=0)
        # print(f"{preds_test.shape=}, {len(preds_test)=}")
        # print(f"{preds_test[0:10]=}")

        # return (pred_train, preds_test)

    def make_new_feature(self, model, model_name, dataset, task, directory):
        kf = Utility().get_kf(task, self.KFoldSplit, self.RANDAOM_STATE)
        (data, label, test) = dataset.get_numpy_data()
        label_number = dataset.get_label_number()

        tmp_name = dataset.get_name()
        preds = []
        preds_test = []
        va_idxes = []
        label_num = dataset.get_label_number()

        if label.ndim == 1:
            split_label = label
        else:
            split_label = label[:, 0]
        y_prob_all = np.zeros(len(data))
        y_test_all = np.zeros(len(test))
        y_true_all = np.zeros(len(data))
        for i, (tr_idx, val_idx) in enumerate(kf.split(data, split_label)):
            k_directory = directory / f"fold_{i + 1}"
            va_idxes.append(val_idx)
            train_data, val_data = data[tr_idx], data[val_idx]
            train_label, val_label = label[tr_idx], label[val_idx]
            new_train = SimpleDataset(
                train_data,
                train_label,
                label_number,
                tmp_name + "_train" + str(i + 1),
            )
            new_val = SimpleDataset(
                val_data,
                val_label,
                label_number,
                tmp_name + "_val" + str(i + 1),
            )
            new_test = SimpleDataset(
                test,
                None,
                label_number,
                tmp_name + "_test" + str(i + 1),
            )

            model_config = self.config["model"][model_name]
            model_config["label_number"] = label_num
            new_model = model(
                model_config, k_directory.as_posix(), new_train, new_val, new_test
            )
            # params = model_config.get("learned_params", {})
            params = new_model.get_model_optimized_params(model_config)
            # print(f"{params=}")
            new_model.learn(i, params)
            new_model.reload()
            pred = new_model.forecast(torch.tensor(val_data.astype(np.float32)))
            # print(f"{pred.shape=}, {val_data.shape=}, {train_data.shape=}")
            preds.append(pred.numpy())
            # print(f"{preds[-1].shape=}, {len(preds)=}, {preds[0:10]=}")
            pred_test = new_model.forecast(torch.tensor(test.astype(np.float32)))

            y_prob_all[val_idx] = pred.numpy().reshape(-1)  # Ensure y_prob_all is 1D
            # y_true_all[val_idx] = val_label[:, 0]
            y_true_all[val_idx] = val_label.reshape(-1)  # Ensure y_true_all is 1D
            y_test_all += pred_test.numpy().reshape(-1)  # Ensure y_test_all is 1D

            # print(f"{pred_test.shape=}, {pred_test[0:10]=}")
            # tmp_ans = pred_test[:, 0]
            # # print(f"{label.shape=}, {label[0:10]=}")
            # print(f"{tmp_ans.shape=}, {tmp_ans[0:10]=}")
            # tmp_ans[tmp_ans < 0] = 0.0
            # tmp_ans[tmp_ans >= 0.5] = 1.0
            # acc = accuracy_score(tmp_ans, label)
            # print(f"Fold {i + 1} Accuracy: {acc:.4f}")
            # print(f"{pred_test.shape=}, {test.shape=}, {train_data.shape=}")
            # tmp_diff = pred_test - pred
            # print(f"{tmp_diff=}")

            # preds_test.append(pred_test.numpy())
            # print(f"{preds_test[-1].shape=}, {len(preds_test)=}, {preds_test[0:10]=}")

        # va_idxes = np.concatenate(va_idxes)
        # preds = np.concatenate(preds, axis=0)
        # order = np.argsort(va_idxes)
        # pred_train = preds[order]
        # print(
        #     f"{preds.shape=}, {va_idxes.shape=}, {len(preds)=}, {pred_train.shape=}, {preds_test[-1].shape=}, {len(preds_test)=}"
        # )
        # print(f"{pred_train[0:10]=}")
        # print(f"{preds_test[0][0:10]=}")
        # print(f"{preds_test[1][0:10]=}")
        # print(f"{preds_test[2][0:10]=}")
        # print(f"{preds_test[3][0:10]=}")
        # print(f"{preds_test[4][0:10]=}")
        # preds_test = np.mean(preds_test, axis=0)
        # print(f"{preds_test.shape=}, {len(preds_test)=}")
        # print(f"{preds_test[0:10]=}")

        final_auc = roc_auc_score(y_true_all, y_prob_all)
        print(f"OOF AUC: {final_auc:.4f}")
        return (y_prob_all, y_test_all / self.KFoldSplit)

    def fit(self):
        self.dataset.load()
        (train, label, test) = self.dataset.get_numpy_data()
        # print(f"{train.shape=}")

        # Fit the model on the training data
        self.model.fit(train, label)

    def predict(self):
        _, _, test = self.dataset.get_numpy_data()
        predictions = self.model.predict(test)
        return predictions
