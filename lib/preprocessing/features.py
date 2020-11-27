import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm


class BOW(object):
    def __init__(self, lowercase, ngram_min, ngram_max, max_features, use_idf):
        super().__init__()
        self.lowercase = lowercase
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.max_features = max_features
        self.use_idf = bool(use_idf)

    def get_repr(self):
        return Pipeline(
            [
                (
                    "bow",
                    CountVectorizer(
                        lowercase=self.lowercase,
                        ngram_range=(
                            self.ngram_min,
                            self.ngram_max,
                        ),
                        analyzer="char_wb",
                        max_features=self.max_features,
                    ),
                ),
                ("tfidf", TfidfTransformer(use_idf=self.use_idf)),
            ]
        )

    def get_config(self):
        return {
            "lowercase": self.lowercase,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "max_features": self.max_features,
            "use_idf": self.use_idf,
        }


class StringKernelPresenceBits(object):
    def __init__(self, strings, ngram_min, ngram_max):
        super().__init__()
        self.strings = strings
        self.ngram = (ngram_min, ngram_max)
        self.ngrams = {}
        self.update_ngrams(self.strings)
        self.test_strings = []

    def __call__(self, xs, ys):
        r = np.zeros((len(xs), len(ys)))
        print(f"Computing string kernel for {len(xs)}x{len(ys)}...")
        for x in tqdm(xs):
            for y in ys:
                idx = int(x[0])
                idy = int(y[0])
                first, second = [
                    (
                        self.strings[i]
                        if i < len(self.strings)
                        else self.test_strings[i - len(self.strings)]
                    )
                    for i in (idx, idy)
                ]
                if len(self.ngrams[first]) > len(self.ngrams[second]):
                    first, second = second, first
                for ngram in self.ngrams[first]:
                    if ngram in self.ngrams[second]:
                        r[idx][idy] += 1

    def update_ngrams(self, strings):
        for string in strings:
            self.ngrams[string] = set({})
            for i in range(len(string) - self.ngram[0]):
                for j in range(self.ngram[0], self.ngram[1] + 1):
                    self.ngrams[string].add(string[i : i + j])

    def prepare_test(self, test_strings):
        self.test_strings = test_strings
        self.update_ngrams(self.test_strings)

    def get_train_features(self):
        features = np.arange(len(self.strings)).reshape(-1, 1)
        return features

    def get_test_features(self):
        features = np.arange(
            len(self.strings), len(self.strings) + len(self.test_strings)
        ).reshape(-1, 1)
        return features
