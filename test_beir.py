from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
from search_array import SearchArraySearch, fields, indexed_columns
from beir.retrieval.evaluation import EvaluateRetrieval


import logging
import pathlib
import os


# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def search_array_results_df(results):
    search_array_results = []
    for query_id, query_results in results.items():
        for doc_id, score in query_results.items():
            search_array_results.append((query_id, doc_id, score))
    return pd.DataFrame(search_array_results, columns=["query_id", "doc_id", "score"]).sort_values(by=["query_id", "score"], ascending=[True, False])


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
    print("BASE CORPUS LOADED")
    for field in fields:
        print(f"Field: {field}")
        model = SearchArraySearch(name=dataset, search_column=field)
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)
        path = f"{dataset}_{field}_results.pkl"
        search_array_results_df(results).to_pickle(path)

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        print(f"NDCG@10: {ndcg}")
        print(f"MAP@100: {_map}")
        print(f"Recall@10: {recall}")
