import gensim
import numpy as np


class Word2VecEmbeddings:
    """Class that manages pretraining of Word2Vec embeddings"""

    def __init__(self, texts, emb_size, min_count=1, iter=10, sg=1):
        """Parameters:
        texts: list of str (cleaned texts)
        emb_size: int (vector embedding size to learn)
        min_count: int (minimum count in corpus for a word not to be ignored)
        iter: int (number of training iterations)
        sg: int 0/1 (use skip grams)
        """
        texts = [t.split() for t in texts]
        self.emb_size = emb_size
        self.model = gensim.models.Word2Vec(
            texts, size=emb_size, min_count=min_count, iter=iter, sg=sg
        )

    def get_emb_matrix(self, text_vectorization):
        """Given a TextVectorization object with word to index mappings,
        generate the embedding matrix
        """
        vocabulary = text_vectorization.get_vocabulary()
        emb_matrix = np.zeros((len(vocabulary), self.emb_size))
        cnt = 0
        for i, w in enumerate(vocabulary):
            try:
                embedding = self.model.wv[w]
                if embedding is not None:
                    emb_matrix[i] = embedding
                    cnt += 1
            except Exception:
                continue
        print(f"{100 * cnt / len(vocabulary)}% coverage from Word2Vec")
        return emb_matrix
