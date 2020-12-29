import argparse
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np

from lib.utils.dataset_utils import load_data
from lib.utils.eval_utils import mae_coordinates
from lib.deep.word.bilstm import BiLSTM

parser = argparse.ArgumentParser(description="Word level deep learning")
parser.add_argument(
    "--config", help="path to configuration file", default="./config_deep_word.json"
)


def main():
    args = parser.parse_args()
    config = {}
    with open(args.config) as f:
        config = json.load(f)

    results_path = f"runs/{datetime.now()}-TEST-deep-{config['method']}-wordlevel"
    os.mkdir(results_path)

    df_train, df_val, df_test = load_data()
    df_results = pd.DataFrame()
    df_results["id"] = df_test.id

    if config["method"] == "regression":
        print("Training model for LAT")

        model_lat = BiLSTM(list(df_train.text), **config["model"])
        model_lat.train(
            list(df_train.text),
            list(df_train.lat),
            list(df_val.text),
            list(df_val.lat),
            **config["training"],
        )

        df_train["predicted_lat"] = model_lat.predict(
            list(df_train.text), config["training"]["batch_size"]
        )
        df_val["predicted_lat"] = model_lat.predict(
            list(df_val.text), config["training"]["batch_size"]
        )
        df_results["lat"] = model_lat.predict(
            list(df_test.text), config["training"]["batch_size"]
        )

        print("Training model for LONG")
        model_long = BiLSTM(list(df_train.text), **config["model"])
        model_long.train(
            list(df_train.text),
            list(df_train.long),
            list(df_val.text),
            list(df_val.long),
            **config["training"],
        )

        df_train["predicted_long"] = model_long.predict(
            list(df_train.text), config["training"]["batch_size"]
        )
        df_val["predicted_long"] = model_long.predict(
            list(df_val.text), config["training"]["batch_size"]
        )
        df_results["long"] = model_long.predict(
            list(df_test.text), config["training"]["batch_size"]
        )

        train_score = mae_coordinates(
            np.column_stack([df_train["lat"], df_train["long"]]),
            np.column_stack([df_train["predicted_lat"], df_train["predicted_long"]]),
        )
        val_score = mae_coordinates(
            np.column_stack([df_val["lat"], df_val["long"]]),
            np.column_stack([df_val["predicted_lat"], df_val["predicted_long"]]),
        )
        print(f"TRAIN: {train_score}")
        print(f"VAL: {val_score}")

    df_results.to_csv(results_path + "/test_results.csv", index=False)


if __name__ == "__main__":
    main()
