from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
from search_array import SearchArraySearch, fields
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


def results_frame(results_file):
    results = pd.read_pickle(results_file)
    ranked = results.sort_values(['query_id', 'score'], ascending=[True, False])
    ranked['rank'] = ranked.groupby('query_id').cumcount() + 1

    corpus, queries, qrels = load_corpus("msmarco")
    qframe = pd.Series(queries).to_frame("query")
    qframe["qrels"] = pd.Series(qrels)
    qframe['qrels'] = qframe['qrels'].apply(dict.keys).apply(list)
    qframe = qframe.explode('qrels')
    qframe = qframe.merge(corpus, how="left", left_on="qrels", right_index=True)
    qframe = qframe.rename(columns={'title': 'target'})
    results = qframe.merge(ranked, left_index=True, right_on="query_id")
    results = results.merge(corpus, left_on="doc_id", right_index=True,
                            how="left", suffixes=("_target", "_result")).drop(columns=["title"])
    target_qrel = results.groupby(['query', 'doc_id', 'rank'])['qrels'].apply(list)
    target_text = results.groupby(['query', 'doc_id', 'rank'])['text_target'].apply(list)

    results = results.groupby(['query', 'doc_id', 'rank']).first()
    results['qrels'] = target_qrel
    results['text_target'] = target_text

    results = results.reset_index().sort_values(['query', 'rank'], ascending=[True, True])
    results['match'] = results.apply(lambda x: x['doc_id'] in x['qrels'], axis=1)
    results['match_rank'] = 1e9
    match_rank = results[results['match']].groupby(['query', 'doc_id'])['rank'].min()
    results = results.set_index(['query', 'doc_id'])
    results['match_rank'] = match_rank
    results['match_rank'] = results['match_rank'].fillna(1e9)

    return results


def run_benchmark():
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


if __name__ == '__main__':
    run_benchmark()
