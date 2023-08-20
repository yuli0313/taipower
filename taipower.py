import pandas as pd
import numpy as np
import sklearn
from autosklearn.regression import AutoSklearnRegressor
import autosklearn.classification
import weather
import os
import datetime

from sklearn.model_selection import train_test_split


def lists_to_dictionary(old, new):
    if len(old) == len(new):
        dictionary = dict(zip(old, new))
        return dictionary
    else:
        print("兩個 lists 的長度不相等。")


def predict(df: pd.DataFrame):
    df = df.values.astype("float32")
    x = df[:, :-1]
    y = df[:, -1]

    # define searchpy
    model = AutoSklearnRegressor(time_left_for_this_task=5 * 60, per_run_time_limit=30, n_jobs=8)
    # perform the search
    model.fit(x, y)

    return model


def fix(x):
    if "," in x:
        return "".join(x.split(","))
    else:
        return x


if __name__ == "__main__":

    df = pd.read_csv("./data/solar_daily.csv")
    
    for i in range(df.shape[0]):
        new_df = 