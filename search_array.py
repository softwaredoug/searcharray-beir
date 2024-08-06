from beir.retrieval.search.base import BaseSearch
from searcharray import SearchArray
import pandas as pd
import numpy as np
import Stemmer
import tqdm

import random
import os
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import perf_counter
from tokenizers import tokenizer_from_str, every_tokenizer_str

every_tokenizer = list(every_tokenizer_str())
every_tokenizer = ["text_{tok}".format(tok=tok) for tok in every_tokenizer]
random.shuffle(every_tokenizer)

stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

DATA_DIR = ".searcharray"
# Ensure datadir exists
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def bm25_search(corpus, query, column):
    tokenizer = corpus[column].array.tokenizer
    query_terms = tokenizer(query)
    scores = np.zeros(len(corpus))
    query_terms = set(query_terms)
    for term in query_terms:
        if term != "_":
            scores += corpus[column].array.score(term)
    return scores


def get_top_k(corpus, scores, top_k):
    """Get top k in format for BEIR."""
    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_scores = scores[top_k_idx]
    top_k_ids = corpus.index[top_k_idx].values

    # Query -> Document IDs -> scores
    results = {doc_id: score for doc_id, score in zip(top_k_ids, top_k_scores)}
    return results


def maybe_add_tok_column(column, corpus, tokenizer, tok_str, data_dir=DATA_DIR):
    new_column = f"{column}_{tok_str}"
    if new_column not in corpus.columns:
        logger.info("*****")
        logger.info("*****")
        logger.info("*****")
        logger.info(f"Tokenizing {new_column}")
        corpus[new_column] = SearchArray.index(corpus[column],
                                               data_dir=data_dir,
                                               tokenizer=tokenizer)
        logger.info("DONE")
        logger.info("*****")
        logger.info("*****")
        logger.info("*****")
    return corpus


# Elasticsearch's default English analyzer
# std tokenizer, posessive, lowercase, stopwords, porter v1
ES_DEFAULT_ENGLISH = "text_Nsp|NNN|ls1"

# Ascii folding, with snowball stemming (porter v2), no stopwords
ASCII_SNOWBALL = "text_asp|NNN|lN2"

# Ascii ws snowball, WS TOK, split on number / case changes, with snowball stemming, no stopwords
ASCII_WS_SNOWBALL = "text_awp|pNN|lN2"

# Ascii ws snowball, WS TOK, split on number / case changes, with snowball stemming, no stopwords
UTF8_WS_SNOWBALL = "text_Nwp|pNN|lN2"

FULL_SNOWBALL = "text_asp|pcn|ls2"
FULL_PORTER = "text_asp|pcn|ls1"
FULL_NOSTEM = "text_asp|pcn|lsN"

# Crazy tokenizers
NO_LOWER_NO_STEM = "text_NsN|NNN|NNN"
NO_LOWER_PORTER1 = "text_NsN|NNN|NN1"
NO_LOWER_PORTER2 = "text_NsN|NNN|NN2"


# Add special fields
fields = [ES_DEFAULT_ENGLISH, ASCII_SNOWBALL, ASCII_WS_SNOWBALL, UTF8_WS_SNOWBALL,
          FULL_SNOWBALL, FULL_PORTER, FULL_NOSTEM, NO_LOWER_NO_STEM, NO_LOWER_PORTER1, NO_LOWER_PORTER2] + every_tokenizer[:10]


def existing_indexed_corpus(data_dir=DATA_DIR, name: Optional[str] = None):
    corpus_path = os.path.join(data_dir, f"{name}_corpus_idx.pkl")
    return corpus_path


def indexed_columns(data_dir=DATA_DIR, name: Optional[str] = None):
    # Load only the columns metadata from the pickle file
    try:
        return pd.read_pickle(existing_indexed_corpus(data_dir, name),
                              columns=None).columns
    except FileNotFoundError:
        return []


class SearchArraySearch(BaseSearch):
    def __init__(self,
                 data_dir=DATA_DIR,
                 name: Optional[str] = None,
                 search_column=ES_DEFAULT_ENGLISH,
                 search_callback=bm25_search):
        self.data_dir = data_dir
        self.name = name
        self.search_column = search_column
        self.tok_str = search_column.split("_")[-1]
        self.source_column = search_column.split("_")[0]
        self.search_callback = search_callback

    def index_corpus(self, corpus):
        corpus_path = existing_indexed_corpus(self.data_dir, self.name)
        try:
            corpus = pd.read_pickle(corpus_path)
            logger.info(f"Corpus Loaded at {corpus_path}")
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
        # corpus[self.search_column].array.posns.clear_cache()
        results = {}
        futures_to_query = {}
        ctr = 0
        logger.info(f"Starting search of {self.search_column}")
        bm25_search(corpus,
                    'how many years did william bradford serve as governor of plymouth colony?',
                    self.search_column)
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


# Actual ES - 2024-07-30 20:23:12 - NDCG@10: 0.2275
# Recreated ES NDCG@10: {'NDCG@1': 0.09857, 'NDCG@3': 0.16236, 'NDCG@5': 0.19144, 'NDCG@10': 0.22255, 'NDCG@100': 0.28212,
