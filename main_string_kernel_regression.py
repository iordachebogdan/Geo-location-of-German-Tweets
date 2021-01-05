"""Entry point for regression task using string kernels"""
import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd

# from sklearn import svm

from lib.classic.regression.nusvr import NuSVR
from lib.preprocessing.features import StringKernelPresenceBits
from lib.utils.dataset_utils import load_data, shuffle_df
from lib.utils.eval_utils import mae_coordinates

parser = argparse.ArgumentParser(description="String kernels for regression")
parser.add_argument(
    "--kernel", help="path to kernel", default="./string-kernel-data/presence_3_5.txt"
)
parser.add_argument(
    "--tweet_ids",
    help="path to ids mapping",
    default="./string-kernel-data/tweet_ids.p",
)
parser.add_argument("--test", action="store_true")


def main():
    args = parser.parse_args()
    # create results directory
    results_path = (
        f'runs/{datetime.now()}-{"TEST-" if args.test else ""}{"string_kernel"}'
        f'-{args.kernel.split("/")[-1]}'
    )
    os.mkdir(results_path)

    # load train, val, test datasets
    df_train, df_val, df_test = load_data()
    if args.test:
        df_train = pd.concat([df_train, df_val])
        df_train = shuffle_df(df_train)
        df_val = df_test

    # features are now just the ids of the tweets
    features_train = np.array([[i] for i in df_train["id"]])
    features_val = np.array([[i] for i in df_val["id"]])

    kernel = StringKernelPresenceBits(args.kernel, args.tweet_ids)

    df_performance = pd.DataFrame()
    for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
        for nu in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # fit regressors with different parameters
            regressor_builder = NuSVR(c, nu, kernel)
            print("Fitting latitude...")
            regressor_lat = regressor_builder.get_regressor()
            # regressor_lat = svm.SVR(kernel=kernel, C=0.0001)
            regressor_lat.fit(features_train, df_train["lat"])
            predict_train_lat = regressor_lat.predict(features_train)
            predict_val_lat = regressor_lat.predict(features_val)
            print("Fitting longitude...")
            regressor_long = regressor_builder.get_regressor()
            # regressor_long = svm.SVR(kernel=kernel, C=0.0001)
            regressor_long.fit(features_train, df_train["long"])
            predict_train_long = regressor_long.predict(features_train)
            predict_val_long = regressor_long.predict(features_val)

            # compute MAE for train and val predictions
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

            # store test predictions
            if args.test:
                df_res = pd.DataFrame()
                df_res["id"] = df_val["id"]
                df_res["lat"] = df_val["predict_lat"]
                df_res["long"] = df_val["predict_long"]
                df_res.to_csv(results_path + "/test_results.csv", index=False)

            # store current configuration performance
            df_performance = df_performance.append(
                {
                    "c": c,
                    "nu": nu,
                    "mae_train": mae_train,
                    "mae_val": mae_val,
                },
                ignore_index=True,
            )
    df_performance.to_csv(results_path + "/results.csv", header=True)


if __name__ == "__main__":
    main()
