from beir.retrieval.search.base import BaseSearch
from searcharray import SearchArray
import pandas as pd
import numpy as np
import Stemmer
import tqdm

import os
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import perf_counter
from tokenizers import every_tokenizer, tokenizer_from_str

stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

DATA_DIR = ".searcharray"
# Ensure datadir exists
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def bm25_search(corpus, query, column):
    tokenizer = corpus[column].array.tokenizer
    query = tokenizer(query)
    scores = np.zeros(len(corpus))
    for q in query:
        scores += corpus[column].array.score(q)
    return scores


def get_top_k(corpus, scores, top_k):
    """Get top k in format for BEIR."""
    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_scores = scores[top_k_idx]
    top_k_ids = corpus.index[top_k_idx].values

    # Query -> Document IDs -> scores
    return {doc_id: score for doc_id, score in zip(top_k_ids, top_k_scores)}


def maybe_add_tok_column(column, corpus, tokenizer, tok_str, data_dir=DATA_DIR):
    new_column = f"{column}_{tok_str}"
    if new_column not in corpus.columns:
        logger.info("*****")
        logger.info("*****")
        logger.info("*****")
        logger.info(f"Tokenizing {new_column}")
        corpus[new_column] = SearchArray.index(corpus[column], data_dir=data_dir, tokenizer=tokenizer)
        logger.info("DONE")
        logger.info("*****")
        logger.info("*****")
        logger.info("*****")
    return corpus


# Elasticsearch's default English analyzer
# std tokenizer, posessive, stopwords, porter v1
ES_DEFAULT_ENGLISH = "text_NsNNlps1"

# Ascii folding, with snowball stemming (porter v2), no stopwords
ASCII_SNOWBALL = "text_asNNlpN2"

# Ascii ws snowball, WS TOK, split on number / case changes, with snowball stemming, no stopwords
ASCII_WS_SNOWBALL = "text_awpNNlpN2"

# Ascii ws snowball, WS TOK, split on number / case changes, with snowball stemming, no stopwords
UTF8_WS_SNOWBALL = "text_NwpNNlpN2"


class SearchArraySearch(BaseSearch):
    def __init__(self, data_dir=None,
                 name: Optional[str] = None,
                 search_column=UTF8_WS_SNOWBALL,
                 search_callback=bm25_search):
        self.data_dir = data_dir
        self.name = name
        self.search_column = search_column
        self.tok_str = search_column.split("_")[-1]
        self.source_column = search_column.split("_")[0]
        self.search_callback = search_callback

    def index_corpus(self, corpus):
        corpus_path = os.path.join(DATA_DIR, f"{self.name}_corpus_idx.pkl")
        try:
            corpus = pd.read_pickle(corpus_path)
            logger.info("Corpus Loaded")
        except FileNotFoundError:
            corpus = pd.DataFrame(corpus)
            corpus.transpose()

        if self.search_column not in corpus.columns:
            if corpus[self.source_column].dtype == 'object':
                corpus[self.source_column].fillna("", inplace=True)
                tok = tokenizer_from_str(self.tok_str)
                orig_columns = corpus.columns
                corpus = maybe_add_tok_column(self.source_column, corpus,
                                              tok,
                                              self.tok_str,
                                              data_dir=self.data_dir)

                if len(orig_columns) != len(corpus.columns) or not os.path.exists(corpus_path):
                    logger.info(f"Saving to {corpus_path}")
                    pd.to_pickle(corpus, corpus_path)
        return corpus

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               *args,
               **kwargs) -> Dict[str, Dict[str, float]]:
        corpus = self.index_corpus(corpus)
        results = {}
        futures_to_query = {}
        ctr = 0
        start = perf_counter()
        with ThreadPoolExecutor() as executor:
            with tqdm.tqdm(total=len(queries)) as pbar:
                for query_id, query in queries.items():
                    futures_to_query[executor.submit(self.search_callback, corpus, query, self.search_column)] = (query_id, query)
                    if len(futures_to_query) > 100 or (ctr == len(queries) - 1):
                        for future in as_completed(futures_to_query):
                            result_query_id, result_query = futures_to_query[future]
                            result = future.result()
                            results[result_query_id] = get_top_k(corpus, result, top_k)
                        elapsed = perf_counter() - start
                        qps = (ctr + 1) / elapsed
                        pbar.set_description(f"Processed {ctr + 1} queries | {qps:.2f} QPS")
                        pbar.update(len(futures_to_query))
                        futures_to_query = {}
                    ctr += 1
        assert len(futures_to_query) == 0
        return results
