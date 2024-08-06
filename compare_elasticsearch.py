import pandas as pd
import numpy as np
import json

es_results = pd.read_pickle('ranking_df.pkl')
search_array_results = pd.read_pickle('msmarco_text_Nsp|NNN|ls1_results.pkl')

compare_query = "what do dark chocolate do for body"


def bm25_search_es(corpus, query, column):
    tokenizer = corpus[column].array.tokenizer
    query_terms = tokenizer(query)
    scores = np.zeros(len(corpus))
    query_terms = set(query_terms)
    for term in query_terms:
        if term != "_":
            corpus[f"{term}_docfreq"] = corpus[column].array.docfreq(term)
            corpus[f"{term}_termfreq"] = corpus[column].array.termfreqs(term)
            scores += corpus[column].array.score(term) * 2.2
            corpus[f'{term}_bm25'] = corpus[column].array.score(term)
            corpus[f'{term}_bm25_es'] = corpus[column].array.score(term) * 2.2

    return scores


expected_tfs = {}
corpus = pd.read_pickle('.searcharray/msmarco_corpus_idx.pkl')


# Pull out all hits from Elasticsearch results, get
# explains for each hit for comparison with searcharray
# Get term freqs into dict for each id
with open('chocolate.json', 'r') as f:
    es_response = json.load(f)
    hits = es_response['hits']['hits']
    for hit in hits:
        doc_id = hit['_id']
        expected_tfs[doc_id] = {}
        term_explains = hit['_explanation']['details'][0]['details']
        for term_explain in term_explains:
            #  Parse out  "weight(txt:chocol in 372905) from description of term_explain
            term = term_explain['description'].split(' ')[0].split(':')[1]
            tf_explain = term_explain['details'][0]['details'][2]
            assert tf_explain['description'].startswith("tf")
            tf_details = tf_explain['details'][0]
            assert tf_details['description'].startswith("freq,")
            tf = tf_details['value']
            expected_tfs[doc_id][term] = tf
            print(term, tf)


corpus['doclens'] = corpus['text_Nsp|NNN|ls1'].array.doclengths()
bm25_scores = bm25_search_es(corpus, compare_query, 'text_Nsp|NNN|ls1')
corpus['bm25_scores'] = bm25_scores
avg_doc_len = corpus['text_Nsp|NNN|ls1'].array.avg_doc_length

for doc_id, tfs in expected_tfs.items():
    for term, expected_tf in tfs.items():
        term_freq = corpus[f'{term}_termfreq'][doc_id]
        expected_tf = expected_tfs[doc_id][term]
        assert term_freq == expected_tf
        print(f"doc_id: {doc_id} Term: {term}, Expected: {expected_tf}, Actual: {term_freq}")


for query_id in es_results['query_id'].unique():
    es_query_results = es_results[es_results['query_id'] == query_id]
    query = es_query_results.iloc[0]['query']
    search_array_query_results = search_array_results[search_array_results['query_id'] == query_id]

    lhs_ids = set()
    rhs_ids = set()
    for lhs_row, rhs_row in zip(es_query_results.iterrows(), search_array_query_results.iterrows()):
        lhs_ids.add(lhs_row[1]['doc_id'])
        rhs_ids.add(rhs_row[1]['doc_id'])
    jaccard = len(lhs_ids.intersection(rhs_ids)) / len(lhs_ids.union(rhs_ids))
    print(query, jaccard)
