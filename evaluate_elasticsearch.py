## Taken from https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/lexical/evaluate_bm25.py
# flake8: noqa
"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch.

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from time import perf_counter

import pathlib, os, random
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="dev")

# Lexical Retrieval using Bm25 (Elasticsearch) ####
# Provide a hostname (localhost) to connect to ES instance
# Define a new index name or use an already existing one.
# We use default ES settings for retrieval
# https://www.elastic.co/

hostname = "localhost"
index_name = dataset

#### Intialize ####
# (1) True - Delete existing index and re-index all documents from scratch
# (2) False - Load existing index
initialize = False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname,
             initialize=initialize, number_of_shards=number_of_shards)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
start = perf_counter()
results = retriever.retrieve(corpus, queries)
end = perf_counter()
logging.info(f"Retrieval Time: {end - start} | QPS: {len(queries)/(end - start)}")

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

import pandas as pd

#### Retrieval Example ####
ranking_df = []
for query_id, scores_dict in results.items():
    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        ranking_df.append({'query_id': query_id,
                           'query': queries[query_id],
                           'doc_id': doc_id,
                           'score': scores[rank][1],
                           'title': corpus[doc_id].get("title"),
                           'text': corpus[doc_id].get("text")})

ranking_df = pd.DataFrame(ranking_df)
ranking_df.to_pickle('ranking_df.pkl')
