import argparse
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline

from lib.classic.classification.svm import SVM

from lib.preprocessing.cleaning import clean
from lib.preprocessing.features import BOW

from lib.utils.classification_utils import (
    ClassificationOnCities,
    ClassificationOnKMeans,
    ClassificationOnRegions,
)
from lib.utils.config_utils import expand_config
from lib.utils.dataset_utils import load_data, shuffle_df
from lib.utils.eval_utils import mae_coordinates, class_accuracy

parser = argparse.ArgumentParser(description="Classic ML algorithms for classification")
parser.add_argument(
    "--config", help="path to configuration file", default="./config.json"
)
parser.add_argument("--test", action="store_true")


def build_representation(config):
    if "bow" in config:
        return BOW(**config["bow"])
    else:
        raise Exception()


def build_pipeline(representation, classifier):
    return Pipeline(
        [("repr", representation.get_repr()), ("regr", classifier.get_classifier())]
    )


def build_classifier(config):
    if "svm" in config:
        return SVM(**config["svm"])
    else:
        raise Exception()


def build_class_logic(config, df_train):
    if "cities" in config:
        return ClassificationOnCities(df_train, **config["cities"])
    elif "kmeans" in config:
        return ClassificationOnKMeans(df_train, **config["kmeans"])
    elif "regions" in config:
        return ClassificationOnRegions(df_train, **config["regions"])


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
        classifier = build_classifier(config["algorithm"])
        pipeline = build_pipeline(representation, classifier)

        df_train, df_val, df_test = load_data()
        if args.test:
            df_train = pd.concat([df_train, df_val])
            df_train = shuffle_df(df_train)
            df_val = df_test

        class_logic = build_class_logic(config["class_logic"], df_train)
        class_logic.set_true_class(df_train)
        if not args.test:
            class_logic.set_true_class(df_val)

        for df in [df_train, df_val]:
            df["clean_text"] = [clean(text) for text in df["text"]]

        print("Fitting classifier...")
        pipeline.fit(df_train["clean_text"], df_train["true_class"])

        df_train["predict_class"] = pipeline.predict(df_train["clean_text"])
        df_val["predict_class"] = pipeline.predict(df_val["clean_text"])

        class_logic.set_predicted_coords(df_train)
        class_logic.set_predicted_coords(df_val)

        true_train = np.column_stack([df_train["lat"], df_train["long"]])
        predict_train = np.column_stack(
            [df_train["predict_lat"], df_train["predict_long"]]
        )
        mae_train = mae_coordinates(true_train, predict_train)
        class_acc_train = class_accuracy(
            df_train["true_class"], df_train["predict_class"]
        )

        mae_val = float("inf")
        class_acc_val = float("inf")
        if not args.test:
            true_val = np.column_stack([df_val["lat"], df_val["long"]])
            predict_val = np.column_stack(
                [df_val["predict_lat"], df_val["predict_long"]]
            )
            mae_val = mae_coordinates(true_val, predict_val)
            class_acc_val = class_accuracy(
                df_val["true_class"], df_val["predict_class"]
            )

        print(f"Class accuracy TRAIN: {class_acc_train}")
        print(f"Class accuracy VALIDATION: {class_acc_val}")
        print(f"Prediction accuracy TRAIN: {mae_train}")
        print(f"Prediction accuracy VALIDATION: {mae_val}")

        if args.test:
            df_res = pd.DataFrame()
            df_res["id"] = df_val["id"]
            df_res["lat"] = df_val["predict_lat"]
            df_res["long"] = df_val["predict_long"]
            df_res.to_csv(current_results_path + "/test_results.csv", index=False)

        df_performance = df_performance.append(
            {
                **representation.get_config(),
                **classifier.get_config(),
                "mae_train": mae_train,
                "mae_val": mae_val,
                "class_acc_train": class_acc_train,
                "class_acc_val": class_acc_val,
            },
            ignore_index=True,
        )

        df_train.to_csv(current_results_path + "/train_results.csv", header=True)
        df_val.to_csv(current_results_path + "/val_results.csv", header=True)

    df_performance.to_csv(results_path + "/results.csv", header=True)


if __name__ == "__main__":
    main()
