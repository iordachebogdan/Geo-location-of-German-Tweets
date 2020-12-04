import pandas as pd
import pickle
import re


def clean(text):
    text = re.sub(r"@[a-zA-Z0-9äöüÄÖÜß]+", " ", text)
    text = re.sub(r"https?://[^ ]+", " ", text)
    text = re.sub(r"www.[^ ]+", " ", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß]", " ", text)
    text = re.sub(" +", " ", text).strip()
    text = text.lower()
    return text


def load_data():
    COLS_LABELED = ["id", "lat", "long", "text"]
    COLS_NOT_LABELED = ["id", "text"]
    df_train = pd.read_csv("data/training.txt", names=COLS_LABELED)
    df_val = pd.read_csv("data/validation.txt", names=COLS_LABELED)
    df_test = pd.read_csv("data/test.txt", names=COLS_NOT_LABELED)

    return df_train, df_val, df_test


df_train, df_val, df_test = load_data()

cnt = 0
id_to_index = {}
with open(
    "string-kernel-data/String_Kernels_Package_v1.0/String_Kernels_Package/code/"
    "tweet_sentences_cleaned.txt",
    "w",
) as f:
    for df in [df_train, df_val, df_test]:
        for _, row in df.iterrows():
            sentence = row["text"]
            # sentence = re.sub(r"\s+", " ", sentence).strip()
            sentence = clean(sentence)
            id_to_index[int(row["id"])] = cnt
            cnt += 1
            f.write(sentence + "\n")

assert len(df_train) + len(df_val) + len(df_test) == cnt
assert len(id_to_index) == cnt

with open(
    "string-kernel-data/String_Kernels_Package_v1.0/String_Kernels_Package/code/"
    "tweet_ids_cleaned.p",
    "wb",
) as f:
    pickle.dump(id_to_index, f)
