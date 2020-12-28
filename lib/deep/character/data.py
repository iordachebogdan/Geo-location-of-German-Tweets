import re
import numpy as np


class Data:
    def __init__(
        self,
        df,
        label_key=None,
        alphabet="abcdefghijklmnopqrstuvwxyzäöüß0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        input_size=500,
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
        str2idx = np.zeros(self.length, dtype="int64")
        pos = 0
        for c in s:
            if c in self.alphabet:
                str2idx[pos] = self.dict[c]
                pos += 1
            elif c == " ":
                pos += 1
            if pos == self.length:
                break
        return str2idx
