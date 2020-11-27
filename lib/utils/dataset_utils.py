import pandas as pd


def load_data():
    COLS_LABELED = ["id", "lat", "long", "text"]
    COLS_NOT_LABELED = ["id", "text"]
    df_train = pd.read_csv("data/training.txt", names=COLS_LABELED)
    df_val = pd.read_csv("data/validation.txt", names=COLS_LABELED)
    df_test = pd.read_csv("data/test.txt", names=COLS_NOT_LABELED)

    return shuffle_df(df_train), df_val, df_test


def shuffle_df(df):
    return df.sample(frac=1).reset_index(drop=True)


def kfold(df_train, df_val, num_folds):
    df = pd.concat([df_train, df_val])
    df = shuffle_df(df)

    num = df.shape[0]
    fold_size = num // num_folds
    for i in range(num_folds):
        df_fold_train = pd.concat(
            [df.iloc[: i * fold_size, :], df.iloc[(i + 1) * fold_size :, :]]
        )
        df_fold_val = df.iloc[i * fold_size : (i + 1) * fold_size, :]
        yield df_fold_train, df_fold_val
