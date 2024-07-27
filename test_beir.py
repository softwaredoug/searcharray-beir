from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
from search_array import SearchArraySearch
from beir.retrieval.evaluation import EvaluateRetrieval


import logging
import pathlib
import os


# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def load_corpus(dataset):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
    corpus = pd.DataFrame(corpus).transpose()
    return corpus, queries, qrels


if __name__ == '__main__':
    dataset = "msmarco"
    corpus, queries, qrels = load_corpus(dataset)
    model = SearchArraySearch(name=dataset)
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print(f"NDCG@10: {ndcg}")
    print(f"MAP@100: {_map}")
    print(f"Recall@10: {recall}")
