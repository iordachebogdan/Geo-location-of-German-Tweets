import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm


class BOW(object):
    def __init__(
        self, lowercase, ngram_min, ngram_max, max_features, use_idf, analyzer
    ):
        super().__init__()
        self.lowercase = lowercase
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.max_features = max_features
        self.use_idf = bool(use_idf)
        self.analyzer = analyzer

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
                        analyzer=self.analyzer,
                        max_features=self.max_features,
                        tokenizer=(
                            None if self.analyzer != "word" else lambda t: t.split(" ")
                        ),
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
            "analyzer": self.analyzer,
        }


class StringKernelPresenceBits(object):
    def __init__(self, kernel_path, ids_path):
        super().__init__()
        self.id_to_index = pickle.load(open(ids_path, "rb"))
        self.kernel_matrix = []
        self.r = None
        cnt = 0
        with open(kernel_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.kernel_matrix.append([int(x) for x in line.strip().split(" ")])
                assert len(self.kernel_matrix[-1]) == len(self.id_to_index)
                cnt += 1
                if cnt % 2000 == 0:
                    print(f"Read {cnt} lines...")
        assert len(self.kernel_matrix) == len(self.id_to_index)

    def __call__(self, xs, ys):
        if xs is ys and self.r is not None:
            print("use cached")
            return self.r

        r = np.zeros((len(xs), len(ys)))
        print(f"Computing string kernel for {len(xs)}x{len(ys)}...")
        for i, x in enumerate(tqdm(xs)):
            for j, y in enumerate(ys):
                idx = self.id_to_index[int(x[0])]
                idy = self.id_to_index[int(y[0])]
                r[i, j] = self.kernel_matrix[idx][idy]
        if xs is ys:
            print("cache kernel matrix")
            self.r = r
        return r
