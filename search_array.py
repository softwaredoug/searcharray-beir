from beir.retrieval.search.base import BaseSearch
from searcharray import SearchArray
import pandas as pd
import numpy as np
import Stemmer
import tqdm

import os
import re
from typing import Dict, Optional
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import perf_counter

stemmer = Stemmer.Stemmer('english', maxCacheSize=0)

DATA_DIR = ".searcharray"
# Ensure datadir exists
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def unnest_list(sublist):
    flattened_list = []
    for item in sublist:
        if isinstance(item, list):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


def stem_word(word):
    return stemmer.stemWord(word)


def remove_posessive(word):
    if word.endswith("'s"):
        return word[:-2]
    return word


def split_on_case_change(s):
    matches = re.finditer(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', s)
    return [m.group(0) for m in matches]


fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


stopwords_list = ["the", "a", "an", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that", "these", "those", "then",
                  "just", "so", "than", "such", "both", "through", "about", "for", "is", "of"]


def snowball_tokenizer(text):
    text = text.translate(all_trans)
    split = text.split()
    sw_removed = []

    for tok in split:
        if tok.lower() in stopwords_list:
            sw_removed.append("")
            continue
        sw_removed.append(tok)

    split = unnest_list([split_on_case_change(tok) for tok in sw_removed])
    return [remove_posessive(stem_word(token.lower()))
            for token in split]


def bm25_search(corpus, query):
    query = snowball_tokenizer(query)
    scores = np.zeros(len(corpus))
    for q in query:
        scores += corpus['text_snowball'].array.score(q)
    return scores


def get_top_k(corpus, scores, top_k):
    """Get top k in format for BEIR."""

    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_scores = scores[top_k_idx]
    top_k_ids = corpus.index[top_k_idx].values

    # Query -> Document IDs -> scores
    return {doc_id: score for doc_id, score in zip(top_k_ids, top_k_scores)}


class SearchArraySearch(BaseSearch):
    def __init__(self, data_dir=None,
                 name: Optional[str] = None,
                 search_callback=bm25_search):
        self.data_dir = data_dir
        self.name = name
        self.search_callback = search_callback

    def index_corpus(self, corpus):
        corpus_path = os.path.join(DATA_DIR, f"{self.name}_corpus_idx.pkl")
        try:
            corpus = pd.read_pickle(corpus_path)
            logger.info("Corpus Loaded")
            return corpus
        except FileNotFoundError:
            corpus = pd.DataFrame(corpus)
            corpus.transpose()
            for column in corpus.columns:
                if corpus[column].dtype == 'object':
                    corpus[column].fillna("", inplace=True)
                    logger.info("*****")
                    logger.info(f"Tokenizing {column}")
                    corpus[f'{column}_snowball'] = SearchArray.index(corpus[column], data_dir=DATA_DIR, tokenizer=snowball_tokenizer)
                    logger.info("*****")
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
                    futures_to_query[executor.submit(self.search_callback, corpus, query)] = (query_id, query)
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
