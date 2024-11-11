from beir.retrieval.search.base import BaseSearch
from searcharray import SearchArray
import pandas as pd
import numpy as np
import tqdm
import cProfile
from sort import get_top_k

import os
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import perf_counter
from lucytok import english
from similarity import bm25_similarity_exp


DATA_DIR = "~/.searcharray"
DATA_DIR = os.path.expanduser(DATA_DIR)
# Ensure datadir exists
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


sum_idfs_cache = {}


def sum_idfs(corpus, search_column):
    """Sum of document frequencies for each term in the given doc_ids."""
    sdf_path = os.path.join(DATA_DIR, f"{search_column}_sum_idf.npy")
    try:
        return sum_idfs_cache[search_column]
    except KeyError:
        try:
            sdfs = np.load(sdf_path)
            sum_idfs_cache[search_column] = sdfs
            return sdfs
        except FileNotFoundError:
            arr = corpus[search_column].array
            sdfs = np.zeros(len(arr), dtype=np.float32)
            logger.info(f"Computing sum of document frequencies for {search_column}")
            num_docs = len(arr)
            for doc_id in tqdm.tqdm(range(num_docs)):
                terms = arr.term_mat[doc_id]
                for term_id in terms[0].cols:
                    dfs = arr.posns.docfreq(term_id)
                    idf = np.log((num_docs + 1) / (dfs + 1)) + 1
                    sdfs[doc_id] += idf
            np.save(sdf_path, sdfs)
            sum_idfs_cache[search_column] = sdfs
            return sdfs


def bm25_search_exp(corpus, query, column):
    tokenizer = corpus[column].array.tokenizer
    query_terms = tokenizer(query)
    scores = None
    query_terms = set(query_terms)

    sum_idf = sum_idfs(corpus, search_column=column)
    sim = bm25_similarity_exp(sum_idf)

    for term in query_terms:
        if term != "_":
            term_scores = corpus[column].array.score(term, similarity=sim)
            if scores is None:
                scores = term_scores
            else:
                scores += term_scores
    if scores is None:
        scores = np.zeros(len(corpus))
    return scores


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def bm25_search(corpus, query, column):
    tokenizer = corpus[column].array.tokenizer
    query_terms = tokenizer(query, flatten=False)
    scores = None
    # query_terms = set(query_terms)
    for term in query_terms:
        if term != "_":
            if isinstance(term, list):
                term = flatten(term)
            term_scores = corpus[column].array.score(term)
            if scores is None:
                scores = term_scores
            else:
                scores += term_scores
    if scores is None:
        scores = np.zeros(len(corpus))
    return scores


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
ES_DEFAULT_ENGLISH = "text_Nsp->NNN->l->sNNN->1"


EVERYTHING = "text_asp->pcn->l->scbp->1"
EVERYTHING_NOSW = "text_asp->pcn->l->Ncbp->1"
EVERYTHING_NO_SW_CP = "text_asp->pcn->l->NNbp->1"
EVERYTHING_NO_SW_CP_BR = "text_asp->pcn->l->NNNp->1"
EVERYTHING_NO_SW_CP_BR_PL = "text_asp->pcn->l->NNNN->1"


# Add special fields
fields = [ES_DEFAULT_ENGLISH,
          EVERYTHING, EVERYTHING_NOSW,
          EVERYTHING_NO_SW_CP, EVERYTHING_NO_SW_CP_BR,
          EVERYTHING_NO_SW_CP_BR_PL]


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
        self.profile = cProfile.Profile()

    def index_corpus(self, corpus):
        corpus_path = existing_indexed_corpus(self.data_dir, self.name)
        try:
            corpus = pd.read_pickle(corpus_path)
            logger.info(f"Corpus Loaded at {corpus_path}")
        except FileNotFoundError:
            corpus = pd.DataFrame(corpus)
            corpus.transpose()
        except EOFError:
            logger.error(f"Corpus at {corpus_path} is corrupted. Rebuilding.")
            corpus = pd.DataFrame(corpus)
            corpus.transpose()

        if self.search_column not in corpus.columns:
            logger.info(f"Reindexing {self.search_column}")
            if corpus[self.source_column].dtype == 'object':
                corpus[self.source_column].fillna("", inplace=True)
                tok = english(self.tok_str)
                orig_columns = corpus.columns
                corpus = maybe_add_tok_column(self.source_column, corpus,
                                              tok,
                                              self.tok_str,
                                              data_dir=self.data_dir)
                assert self.search_column in corpus.columns

                if len(orig_columns) != len(corpus.columns) or not os.path.exists(corpus_path):
                    logger.info(f"Saving to {corpus_path}")
                    pd.to_pickle(corpus, corpus_path)
        return corpus

    def _search_single_threaded(self, corpus, queries, top_k):
        start = perf_counter()
        results = {}
        ctr = 0
        corpus[self.search_column].array.posns.cache_gt_then = 10
        with tqdm.tqdm(total=len(queries)) as pbar:
            for query_id, query in queries.items():
                result_query = self.search_callback(corpus, query, self.search_column)
                results[query_id] = get_top_k(corpus, result_query, top_k)
                elapsed = perf_counter() - start
                qps = (ctr + 1) / elapsed
                pbar.set_description(f"Processed {ctr + 1} queries | {qps:.2f} QPS")
                ctr += 1
                pbar.update(1)
        return results

    def _search_multi_threaded(self, corpus, queries, top_k):
        start = perf_counter()
        results = {}
        futures_to_query = {}
        ctr = 0
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

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               *args,
               **kwargs) -> Dict[str, Dict[str, float]]:
        corpus = self.index_corpus(corpus)
        # corpus[self.search_column].array.posns.clear_cache()
        logger.info(f"Starting search of {self.search_column}")
        # sum_idfs(corpus, search_column=self.search_column)
        # results = self.profile.runcall(self._search_multi_threaded, corpus, queries, top_k)
        results = self.profile.runcall(self._search_single_threaded, corpus, queries, top_k)
        self.profile.dump_stats("search.prof")
        return results


# Actual ES - 2024-07-30 20:23:12 - NDCG@10: 0.2275
# Recreated ES NDCG@10: {'NDCG@1': 0.09857, 'NDCG@3': 0.16236, 'NDCG@5': 0.19144, 'NDCG@10': 0.22255, 'NDCG@100': 0.28212,
