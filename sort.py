import numpy as np
import pandas as pd
from time import perf_counter


def get_top_k_orig(corpus, scores, top_k):
    """Get top k in format for BEIR."""
    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_scores = scores[top_k_idx]
    top_k_ids = corpus.index[top_k_idx].values

    # Query -> Document IDs -> scores
    results = {doc_id: score for doc_id, score in zip(top_k_ids, top_k_scores)}
    return results


def get_top_k_zeros_only(corpus, scores, top_k):
    """Get top k, but only argpartition zeros."""
    non_zero_idxs = np.argwhere(scores != 0).flatten()
    if len(non_zero_idxs) == 0:
        # Just return first top_k
        top_k_scores = scores[:top_k]
        top_k_ids = corpus.index[:top_k].values
        return {doc_id: float(score) for doc_id, score in zip(top_k_ids, top_k_scores)}
    elif len(non_zero_idxs) < top_k:
        scores = scores[non_zero_idxs]
        top_k_scores = scores
        top_k_ids = corpus.index[non_zero_idxs].values
        return {doc_id: float(score) for doc_id, score in zip(top_k_ids, top_k_scores)}
    else:
        scores = scores[non_zero_idxs]
        idx_into_non_zero = np.argpartition(scores, -top_k)[-top_k:]
        top_k_scores = scores[idx_into_non_zero]
        top_k_ids = corpus.index[non_zero_idxs[idx_into_non_zero]].values
        return {doc_id: float(score) for doc_id, score in zip(top_k_ids, top_k_scores)}


def random_1m_array(set_k=10000):
    random_mask = np.random.choice([True, False], size=1000000, p=[set_k / 1000000, 1 - set_k / 1000000])
    arr = np.zeros(1000000)
    arr[random_mask] = np.random.rand(np.sum(random_mask))
    return arr


def test_get_top_k(times=1000):
    df = pd.DataFrame({"text": ["a"] * 1000000}, index=range(1000000))
    start = perf_counter()
    for i in range(times):
        scores = random_1m_array()
        all_zeros = np.zeros(1000000)
        for k in [10, 100, 1000, 10000]:
            results1 = get_top_k_orig(df, scores, k)
            results2 = get_top_k_zeros_only(df, scores, k)
            if len(results2) == len(results1):
                assert results1 == results2
            # FOR SOME REASON ALL 0s SLOWER
            # result_zeros = get_top_k_zeros_only(df, all_zeros, k)
            # assert len(result_zeros) == k
    end = perf_counter()
    return end - start


get_top_k = get_top_k_zeros_only


if __name__ == "__main__":
    print(test_get_top_k())
