import argparse
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline

from lib.classic.regression.nusvr import NuSVR
from lib.classic.regression.linearsvr import LinearSVR

from lib.preprocessing.cleaning import basic_clean
from lib.preprocessing.features import BOW

from lib.utils.config_utils import expand_config
from lib.utils.dataset_utils import load_data, shuffle_df
from lib.utils.eval_utils import mae_coordinates

parser = argparse.ArgumentParser(description="Classic ML algorithms for regression")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)
parser.add_argument("--test", action="store_true")


def build_representation(config):
    if "bow" in config:
        return BOW(**config["bow"])
    else:
        raise Exception()


def build_regressor(config):
    if "nusvr" in config:
        return NuSVR(**config["nusvr"])
    elif "linearsvr" in config:
        return LinearSVR(**config["linearsvr"])
    else:
        raise Exception()


def build_pipeline(representation, regressor):
    return Pipeline(
        [("repr", representation.get_repr()), ("regr", regressor.get_regressor())]
    )


def main():
    args = parser.parse_args()
    config = {}
    with open(args.config) as f:
        config = json.load(f)

    configs = expand_config(config)
    results_path = (
        f'runs/{datetime.now()}-{"TEST-" if args.test else ""}{config["type"]}'
        f'-{config["method"]}-{list(config["algorithm"].keys())[0]}'
    )
    os.mkdir(results_path)
    df_performance = pd.DataFrame()
    for i, config in enumerate(configs):
        print(f"Running {i+1}/{len(configs)}")
        current_results_path = os.path.join(results_path, str(i))
        os.mkdir(current_results_path)
        with open(current_results_path + "/config.json", "w") as f:
            json.dump(config, f, indent=4)

        representation = build_representation(config["representation"])
        regressor = build_regressor(config["algorithm"])
        pipeline_lat = build_pipeline(representation, regressor)
        pipeline_long = build_pipeline(representation, regressor)

        df_train, df_val, df_test = load_data()
        if args.test:
            df_train = pd.concat([df_train, df_val])
            df_train = shuffle_df(df_train)
            df_val = df_test

        for df in [df_train, df_val]:
            df["clean_text"] = [basic_clean(text) for text in df["text"]]

        print("Fitting latitude...")
        pipeline_lat.fit(df_train["clean_text"], df_train["lat"])
        print("Fitting longitude...")
        pipeline_long.fit(df_train["clean_text"], df_train["long"])

        predict_train_lat = pipeline_lat.predict(df_train["clean_text"])
        predict_train_long = pipeline_long.predict(df_train["clean_text"])

        predict_val_lat = pipeline_lat.predict(df_val["clean_text"])
        predict_val_long = pipeline_long.predict(df_val["clean_text"])

        true_train = np.column_stack([df_train["lat"], df_train["long"]])
        predict_train = np.column_stack([predict_train_lat, predict_train_long])
        mae_train = mae_coordinates(true_train, predict_train)

        mae_val = float("inf")
        if not args.test:
            true_val = np.column_stack([df_val["lat"], df_val["long"]])
            predict_val = np.column_stack([predict_val_lat, predict_val_long])
            mae_val = mae_coordinates(true_val, predict_val)

        print(f"Prediction accuracy TRAIN: {mae_train}")
        print(f"Prediction accuracy VALIDATION: {mae_val}")

        df_train["predict_lat"] = predict_train_lat
        df_train["predict_long"] = predict_train_long
        df_val["predict_lat"] = predict_val_lat
        df_val["predict_long"] = predict_val_long

        if args.test:
            df_res = pd.DataFrame()
            df_res["id"] = df_val["id"]
            df_res["lat"] = df_val["predict_lat"]
            df_res["long"] = df_val["predict_long"]
            df_res.to_csv(current_results_path + "/test_results.csv", index=False)

        df_performance = df_performance.append(
            {
                **representation.get_config(),
                **regressor.get_config(),
                "mae_train": mae_train,
                "mae_val": mae_val,
            },
            ignore_index=True,
        )

        df_train.to_csv(current_results_path + "/train_results.csv", header=True)
        df_val.to_csv(current_results_path + "/val_results.csv", header=True)

    df_performance.to_csv(results_path + "/results.csv", header=True)


if __name__ == "__main__":
    main()
