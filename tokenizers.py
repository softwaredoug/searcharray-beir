from typing import List
import regex as re
import os
import requests
from sys import argv
import Stemmer

stemmer = Stemmer.Stemmer("english")


# get es_url from env
es_url = os.getenv("ES_URL")


def run_es_analyzer(text: str) -> List[str]:
    """Run the Elasticsearch English analyzer on the text."""
    # Define the request body
    body = {
        "text": text,
        "analyzer": "english"
    }
    response = requests.get(es_url, json=body)
    return [tok['token'] for tok in response.json()["tokens"]]


# a, an, and, are, as, at, be, but, by, for, if, in, into, is, it, no, not, of, on, or, such, that, the, their, then, there, these, they, this, to, was, will, with
elasticsearch_english_stopwords = [
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with"]


def standard_tokenizer(text: str) -> List[str]:
    """Tokenize text using a standard tokenizer."""
    pattern = r"\w\p{Extended_Pictographic}\p{WB:RegionalIndicator}"

    # Compile the regex pattern
    segment = re.compile(rf"[{pattern}](?:\B\S)*", flags=re.WORD)

    # Find all tokens based on the word boundary pattern
    return segment.findall(text)


def remove_posessive_suffixes(tokens: List[str]) -> List[str]:
    """Remove posessive suffixes from tokens."""

    # As of 3.6, U+2019 RIGHT SINGLE QUOTATION MARK and U+FF07 FULLWIDTH APOSTROPHE are also treated as quotation marks.
    return [re.sub(r"['’]s$", "", token) for token in tokens]

# "rebuilt_english": {
#          "tokenizer":  "standard",
#          "filter": [
#            "english_possessive_stemmer",
#            "lowercase",
#            "english_stop",
#            "english_keywords",
#            "english_stemmer"
#          ]
#        }


def elasticsearch_english(text: str) -> List[str]:
    """Recreate Elasticsearch's English analyzer."""
    # Define the regex pattern for Unicode word boundaries
    tokens = standard_tokenizer(text)
    tokens = remove_posessive_suffixes(tokens)
    # Lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in elasticsearch_english_stopwords]
    # Stem tokens
    tokens = [stemmer.stemWord(token) for token in tokens]
    return tokens


if __name__ == "__main__":
    text = argv[1]
    from_es = run_es_analyzer(text)
    local = elasticsearch_english(text)
    try:
        assert from_es == local, f"Expected {from_es}, but got {local}"
    except AssertionError:
        for term1, term2 in zip(from_es, local):
            if term1 != term2:
                print(f"Expected {term1}, but got {term2}")
