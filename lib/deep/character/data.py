import re
import numpy as np


class Data:
    def __init__(
        self,
        df,
        label_key=None,
        alphabet="abcdefghijklmnopqrstuvwxyzäöüß0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        input_size=750,
        num_of_classes=1,
    ):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.num_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.df = df
        self.label_key = label_key

    def get_data(self):
        batch_indices = []
        one_hot = np.eye(self.num_of_classes, dtype="int64")
        labels = []
        if self.label_key is None:
            for s in self.df.text:
                batch_indices.append(self.str_to_idx(s))
            return np.asarray(batch_indices, dtype="int64")

        for label, s in zip(self.df[self.label_key], self.df.text):
            batch_indices.append(self.str_to_idx(s))
            if self.num_of_classes > 1:
                label = int(label)
                labels.append(one_hot[label])
            else:
                labels.append(label)
        return np.asarray(batch_indices, dtype="int64"), np.asarray(labels)

    def str_to_idx(self, s):
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype="int64")
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
