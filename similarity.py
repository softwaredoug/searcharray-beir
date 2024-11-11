from numpy.typing import NDArray
from searcharray.bm25 import bm25_score
import numpy as np
from searcharray.similarity import Similarity, compute_idf


def bm25_similarity_exp(sum_idfs, k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity function, as in Lucene 9."""
    avg_sum_idfs = np.mean(sum_idfs)

    def bm25(term_freqs: NDArray[np.float32],
             doc_freqs: NDArray[np.float32],
             doc_lens: NDArray[np.float32],
             avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate BM25 scores."""
        if avg_doc_lens == 0:
            return np.zeros_like(term_freqs)
        idf = compute_idf(num_docs, doc_freqs)
        bm25_score(term_freqs,
                   sum_idfs * doc_lens,
                   avg_sum_idfs * avg_doc_lens,
                   idf, k1, b)
        return term_freqs
    return bm25
