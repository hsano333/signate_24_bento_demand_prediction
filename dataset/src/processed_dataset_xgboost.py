# from enum import auto
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from common.my_enum import MLTask
import joblib

import torch
import numpy as np
import pandas as pd
import os


class ProcessedDatasetXGBoost(Dataset):
    def __init__(self, config, name="processed_dataset", time_model=None):
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        self.TRAIN_PATH = os.path.join(current_dir, "data", "original", "train.csv")
        self.TEST_PATH = os.path.join(current_dir, "data", "original", "test.csv")
        self.PREPOCESSED_PATH = os.path.join(
            current_dir, "data", "original", "dataset_preprocessed.pkl"
        )

        self.dateset_name = name
        self.config = config
        self.target_feature = self.config["target_feature"]
        self.time_model = time_model

    def get_numpy_data(self):
        return (self.data.numpy(), self.label.numpy(), self.test_data.numpy())

    def get_test_id(self):
        return self.test_id

    def set_raw_label(self, raw_label):
        # pandas
        self.raw_label = raw_label

    def get_raw_label(self):
        # pandas
        return self.raw_label

    def convert_np_to_torch(self, data):
        return torch.tensor(data.values.astype(np.float32))

    def convert_onehot_encoder(self, encoder, data, name):
        encoderd_data = encoder.transform(data[[name]])
        feature_names = encoder.get_feature_names_out([name])
        tmp_df = pd.DataFrame(encoderd_data, columns=feature_names)
        return pd.concat([data, tmp_df], axis=1)

    def convert_oneshot(self, data, name):
        encoded_data = pd.get_dummies(
            data,
            columns=[
                name,
            ],
        )
        return encoded_data

    def transform_label(self, task):
        if task == MLTask.Regression:
            self.label_number = self.label.shape[0]
            # self.label = torch.from_numpy(self.label)
        else:
            unique = self.raw_label.unique()
            self.le = LabelEncoder()
            raw_np_label = self.le.fit_transform(self.label)
            self.label = torch.from_numpy(self.label)
            self.label_number = unique.shape[0]

    def make_date_id(self):
        start_date = "2013-10-28"
        end_date = "2014-12-01"

        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(all_dates, columns=["date"])
        df["datetime"] = pd.to_datetime(df["date"])
        df["weekday"] = df["datetime"].dt.day_name()
        # df["weekday_i"] = df["date"].dt.dayofweek

        df = df.reset_index()
        df = df.rename(columns={"index": "id"})
        df["week_id"] = df["id"] // 7
        df = df.drop(columns=["id", "date"])

        return df

    def mypreprocessing(self, data, onehot_encoder):
        data["payday"] = data["payday"].fillna(0)
        data["event"] = data["event"].fillna("なし")
        data["remarks"] = data["remarks"].fillna("なし")
        data["precipitation"] = data["precipitation"].apply(
            lambda x: -1 if x == "--" else float(x)
        )
        # data["month"] = data["datetime"].apply(lambda x: int(x.split("-")[1]))
        data["curry"] = data["name"].apply(lambda x: 1 if x.find("カレー") >= 0 else 0)
        data["fun"] = data["remarks"].apply(
            lambda x: 1 if x == "お楽しみメニュー" else 0
        )
        data.loc[data["weather"] == "快晴", "weather"] = "晴れ"
        data.loc[data["weather"] == "雨", "weather"] = "なし"
        data.loc[data["weather"] == "雪", "weather"] = "なし"
        data.loc[data["weather"] == "雷電", "weather"] = "なし"

        data.loc[
            (
                (data["remarks"] != "料理長のこだわりメニュー")
                & (data["remarks"] != "お楽しみメニュー")
            ),
            "remarks",
        ] = "なし"

        data["datetime"] = pd.to_datetime(data["datetime"])

        data = self.convert_oneshot(data, "event")
        data = self.convert_oneshot(data, "remarks")
        data = self.convert_oneshot(data, "weather")
        # data = self.convert_oneshot(data, "week")
        data = self.convert_oneshot(data, "weekday")
        # data = self.convert_oneshot(data, "default")
        # data = self.convert_oneshot(data, "job")
        # data = self.convert_oneshot(data, "marital")
        # data = self.convert_oneshot(data, "education")
        # data = self.convert_oneshot(data, "housing")
        # data = self.convert_oneshot(data, "loan")
        # data = self.convert_oneshot(data, "contact")
        # data = self.convert_oneshot(data, "pdays_dummy")

        data = data.drop(
            [
                # "id",
                "kcal",
                "temperature",
                "week",
                "datetime",
                "name",
                # "poutcome",
                # "age_dummy",
                # "campaign_dummy",
                # "duration_dummy",
                # "previous_dummy",
                # "duration_contact",
            ],
            axis=1,
        )
        return data.astype(float)

    def load(self):
        if os.path.exists(self.PREPOCESSED_PATH) and self.config["overwrite"] is False:
            print(f"Loading preprocessed dataset from {self.PREPOCESSED_PATH}")
            loaded_data = joblib.load(self.PREPOCESSED_PATH)
            self.data = loaded_data["data"]
            self.label = loaded_data["label"]
            self.label_number = self.label.shape[1]
            self.test_data = loaded_data["test_data"]
            self.test_id = loaded_data["datetime"]
            self.raw_label = loaded_data["raw_label"]
            self.target_feature = loaded_data["target_feature"]
            return

        train = pd.read_csv(self.TRAIN_PATH)
        test = pd.read_csv(self.TEST_PATH)

        self.test_id = test["datetime"]
        self.raw_label = train[self.target_feature]

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        onehot_encoder = ohe.fit(train[["name"]])

        train["datetime"] = pd.to_datetime(train["datetime"])
        test["datetime"] = pd.to_datetime(test["datetime"])
        date_df = self.make_date_id()
        train = pd.merge(train, date_df, on="datetime", how="left")
        test = pd.merge(test, date_df, on="datetime", how="left")

        train["y_avg"] = self.time_model.forecast(
            torch.tensor(train[["week_id"]].values)
        )
        # print(f"{train['y_avg'][0:40]=}")
        # print(f"before:{train['y'][0:35]=}")
        # train["y"] = (train["y"] - train["y_avg"]) / train["y_avg"]
        train["y"] = train["y"] - train["y_avg"]
        # print(f"after:{train['y'][0:35]=}")
        train = train.drop(columns=["y_avg"])

        train = self.mypreprocessing(train, onehot_encoder)
        test = self.mypreprocessing(test, onehot_encoder)
        tmp_label = train[self.target_feature]
        train = train.drop([self.target_feature], axis=1)

        train_scaled = pd.DataFrame(
            # scaler.fit_transform(train),
            train,
            columns=train.columns,
            index=train.index,
        )
        test_scaled = pd.DataFrame(
            # scaler.transform(test),
            test,
            columns=test.columns,
            index=test.index,
        )

        self.data = self.convert_np_to_torch(train_scaled)
        self.label = self.convert_np_to_torch(tmp_label).unsqueeze(
            1
        )  # Add a dimension for label
        self.label_number = 1
        self.test_data = self.convert_np_to_torch(test_scaled)

        data_to_save = {
            "data": self.data,
            "label": self.label,
            "test_data": self.test_data,
            "id": self.test_id,
            "raw_label": self.raw_label,
            "target_feature": self.target_feature,
        }
        joblib.dump(data_to_save, self.PREPOCESSED_PATH)
        print(f"Preprocessed dataset saved to {self.PREPOCESSED_PATH}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        return (self.data[ndx], self.label[ndx])

    def get_name(self):
        return self.dateset_name

    def get_column_number(self):
        return self.data.shape[1]

    def get_label_number(self):
        return self.label_number
