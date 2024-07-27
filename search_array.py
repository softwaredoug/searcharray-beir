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
from tokenizers import elasticsearchporter1_tokenizer, elasticsearchsnowball_tokenizer

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


def remove_posessive(text):
    text_without_posesession = []
    for word in text.split():
        if word.endswith("'s"):
            text_without_posesession.append(word[:-2])
        else:
            text_without_posesession.append(word)
    return " ".join(text_without_posesession)


def split_on_case_change(s):
    matches = re.finditer(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', s)
    return [m.group(0) for m in matches]


def split_on_char_num_change(s):
    matches = re.finditer(r'.+?(?:(?<=\d)(?=\D)|(?<=\D)(?=\d)|$)', s)
    return [m.group(0) for m in matches]


fold_to_ascii = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
punct_trans = str.maketrans({key: ' ' for key in string.punctuation})
all_trans = {**fold_to_ascii, **punct_trans}


elasticsearch_stopwords = ["a", "an", "and", "are", "as", "at", "be", "but", "by",
                           "for", "if", "in", "into", "is", "it",
                           "no", "not", "of", "on", "or", "such",
                           "that", "the", "their", "then", "there", "these",
                           "they", "this", "to", "was", "will", "with"]


def snowball_tokenizer(text):
    text = remove_posessive(text)
    return snowball_tokenizer_noposs(text)


def dumbsnowball_tokenizer(text):
    text = text.translate(all_trans)
    split = text.split()
    return [stem_word(token.lower())
            for token in split]


def dumbsnowballnoposs_tokenizer_noposs(text):
    text = remove_posessive(text)
    return dumbsnowball_tokenizer(text)


def snowball_tokenizer_noposs(text):
    text = text.translate(all_trans)
    split = text.split()
    sw_removed = []

    for tok in split:
        if tok.lower() in elasticsearch_stopwords:
            sw_removed.append("")
            continue
        sw_removed.append(tok)

    return [stem_word(token.lower())
            for token in sw_removed]


def ws_tokenizer(text):
    text = text.translate(all_trans)
    split = text.split()
    split = unnest_list([split_on_case_change(tok) for tok in split])
    split = unnest_list([split_on_char_num_change(tok) for tok in split])
    return [token.lower() for token in split]


def ws_tokenizer_noposs(text):
    text = remove_posessive(text)
    return ws_tokenizer(text)


def bm25_search(corpus, query):
    query = snowball_tokenizer_noposs(query)
    scores = np.zeros(len(corpus))
    for q in query:
        scores += corpus['text_snowball_noposs'].array.score(q)
    return scores


def get_top_k(corpus, scores, top_k):
    """Get top k in format for BEIR."""

    top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_k_scores = scores[top_k_idx]
    top_k_ids = corpus.index[top_k_idx].values

    # Query -> Document IDs -> scores
    return {doc_id: score for doc_id, score in zip(top_k_ids, top_k_scores)}


def maybe_add_tok_column(column, corpus, tokenizer, data_dir=DATA_DIR):
    tokenizer_name = tokenizer.__name__
    tokenizer_name = tokenizer_name.split("_")[0]
    new_column = f"{column}_{tokenizer_name}"
    if new_column not in corpus.columns:
        logger.info("*****")
        logger.info(f"Tokenizing {new_column}")
        corpus[new_column] = SearchArray.index(corpus[column], data_dir=data_dir, tokenizer=tokenizer)
    return corpus


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
        except FileNotFoundError:
            corpus = pd.DataFrame(corpus)
            corpus.transpose()

        orig_columns = corpus.columns

        for column in corpus.columns:
            if corpus[column].dtype == 'object':
                corpus[column].fillna("", inplace=True)
                for tokenizer in [snowball_tokenizer, ws_tokenizer,
                                  elasticsearchporter1_tokenizer, elasticsearchsnowball_tokenizer]:
                    corpus = maybe_add_tok_column(column, corpus, tokenizer, data_dir=self.data_dir)

        if len(orig_columns) != len(corpus.columns) or not os.path.exists(corpus_path):
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
                        logger.info(f"query: {query_id} | tok_ws | {ws_tokenizer(query)}")
                        logger.info(f"query: {query_id} | tok_sn | {snowball_tokenizer(query)}")
                        pbar.update(len(futures_to_query))
                        futures_to_query = {}
                    ctr += 1
        assert len(futures_to_query) == 0
        return results
