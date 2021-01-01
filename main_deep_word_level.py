import argparse
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np

from lib.utils.dataset_utils import load_data
from lib.utils.eval_utils import mae_coordinates
from lib.deep.word.bilstm import BiLSTM
from lib.deep.word.word_embeddings import Word2VecEmbeddings
from lib.preprocessing.cleaning import clean
from lib.utils.classification_utils import (
    ClassificationOnKMeans,
    class_labels_to_onehot,
)


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

    df_train.text = [clean(t) for t in df_train.text]
    df_val.text = [clean(t) for t in df_val.text]
    df_test.text = [clean(t) for t in df_test.text]

    word_embeddings = None
    if "word2vec" in config:
        word_embeddings = Word2VecEmbeddings(
            list(df_train.text) + list(df_val.text) + list(df_test.text),
            **config["word2vec"],
        )

    if config["method"] == "regression":
        print("Training model for LAT")

        model_lat = BiLSTM(
            list(df_train.text), word_embeddings=word_embeddings, **config["model"]
        )
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
        model_long = BiLSTM(
            list(df_train.text), word_embeddings=word_embeddings, **config["model"]
        )
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
    else:
        if "kmeans" in config["classification"]:
            num_classes = config["classification"]["kmeans"]["num_classes"]
            cokm = ClassificationOnKMeans(df_train, num_classes)

            train_text = list(df_train.text)
            val_text = list(df_val.text)
            test_text = list(df_test.text)

            cokm.set_true_class(df_train)
            cokm.set_true_class(df_val)

            train_labels = class_labels_to_onehot(
                list(df_train.true_class), num_classes
            )
            val_labels = class_labels_to_onehot(list(df_val.true_class), num_classes)
        else:
            raise Exception("Method not implemented")

        model = BiLSTM(train_text, word_embeddings=word_embeddings, **config["model"])
        model.train(
            train_text,
            train_labels,
            val_text,
            val_labels,
            **config["training"],
        )

        train_predicted_classes = model.predict(
            train_text, config["training"]["batch_size"]
        )
        train_predicted_classes = np.argmax(train_predicted_classes, axis=1)
        val_predicted_classes = model.predict(
            val_text, config["training"]["batch_size"]
        )
        val_predicted_classes = np.argmax(val_predicted_classes, axis=1)
        test_predicted_classes = model.predict(
            test_text, config["training"]["batch_size"]
        )
        test_predicted_classes = np.argmax(test_predicted_classes, axis=1)

        df_train["predict_class"] = train_predicted_classes
        df_val["predict_class"] = val_predicted_classes
        df_test["predict_class"] = test_predicted_classes

        cokm.set_predicted_coords(df_train)
        cokm.set_predicted_coords(df_val)
        cokm.set_predicted_coords(df_test)

        train_score = mae_coordinates(
            np.column_stack([df_train["lat"], df_train["long"]]),
            np.column_stack([df_train["predict_lat"], df_train["predict_long"]]),
        )
        val_score = mae_coordinates(
            np.column_stack([df_val["lat"], df_val["long"]]),
            np.column_stack([df_val["predict_lat"], df_val["predict_long"]]),
        )
        print(f"TRAIN: {train_score}")
        print(f"VAL: {val_score}")

        df_results["lat"] = df_test["predict_lat"]
        df_results["long"] = df_test["predict_long"]

    df_results.to_csv(results_path + "/test_results.csv", index=False)


if __name__ == "__main__":
    main()
