import argparse
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np

from lib.utils.dataset_utils import load_data
from lib.utils.eval_utils import mae_coordinates
from lib.deep.character.data import Data
from lib.deep.character.model import CharacterLevelCNN

parser = argparse.ArgumentParser(description="Character level CNN")
parser.add_argument(
    "--config", help="path to configuration file", default="./config_charcnn.json"
)


def main():
    args = parser.parse_args()
    config = {}
    with open(args.config) as f:
        config = json.load(f)

    results_path = f"runs/{datetime.now()}-TEST-deep-{config['method']}-charcnn"
    os.mkdir(results_path)

    df_train, df_val, df_test = load_data()
    df_results = pd.DataFrame()
    df_results["id"] = df_test.id
    if config["method"] == "regression":
        train_data_lat = Data(df_train, label_key="lat")
        train_data_long = Data(df_train, label_key="long")
        val_data_lat = Data(df_val, label_key="lat")
        val_data_long = Data(df_val, label_key="long")
        test_data = Data(df_test)

        print("Training model for LAT")

        model_lat = CharacterLevelCNN(
            input_size=train_data_lat.length,
            alphabet_size=train_data_lat.alphabet_size,
            num_of_classes=1,
            **config["model"],
        )

        training_inputs, training_labels = train_data_lat.get_data()
        validation_inputs, validation_labels = val_data_lat.get_data()
        model_lat.train(
            training_inputs,
            training_labels,
            validation_inputs,
            validation_labels,
            **config["training"],
        )

        df_train["predicted_lat"] = model_lat.predict(
            training_inputs, config["training"]["batch_size"]
        )
        df_val["predicted_lat"] = model_lat.predict(
            validation_inputs, config["training"]["batch_size"]
        )

        test_inputs = test_data.get_data()
        df_results["lat"] = model_lat.predict(
            test_inputs, config["training"]["batch_size"]
        )

        print("Training model for LONG")

        model_long = CharacterLevelCNN(
            input_size=train_data_long.length,
            alphabet_size=train_data_long.alphabet_size,
            num_of_classes=1,
            **config["model"],
        )

        training_inputs, training_labels = train_data_long.get_data()
        validation_inputs, validation_labels = val_data_long.get_data()
        model_long.train(
            training_inputs,
            training_labels,
            validation_inputs,
            validation_labels,
            **config["training"],
        )

        df_train["predicted_long"] = model_long.predict(
            training_inputs, config["training"]["batch_size"]
        )
        df_val["predicted_long"] = model_long.predict(
            validation_inputs, config["training"]["batch_size"]
        )

        test_inputs = test_data.get_data()
        df_results["long"] = model_long.predict(
            test_inputs, config["training"]["batch_size"]
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
