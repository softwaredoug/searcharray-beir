from typing import List, Optional
import regex as re
import os
import string
import requests
import Stemmer
from porter import PorterStemmer
from functools import partial
import unicodedata
from ascii_fold import unicode_to_ascii


stemmer = Stemmer.Stemmer("english")
porterv1 = PorterStemmer()


punct_trans = str.maketrans({key: ' ' for key in string.punctuation})


# get es_url from env
es_url = os.getenv("ES_URL")


def unnest_list(sublist):
    flattened_list = []
    for item in sublist:
        if isinstance(item, list):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list


def porter2_stem_word(word):
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


std_pattern = r"\w\p{Extended_Pictographic}\p{WB:RegionalIndicator}"
segment = re.compile(rf"[{std_pattern}](?:\B\S)*", flags=re.WORD)


def standard_tokenizer(text: str) -> List[str]:
    """Tokenize text using a standard tokenizer."""
    # Find all tokens based on the word boundary pattern
    return segment.findall(text)


possessive_suffix_regex = re.compile(r"['’]s$")


def remove_posessive_suffixes(tokens: List[str]) -> List[str]:
    """Remove posessive suffixes from tokens."""

    def remove_suffix(token: str) -> str:
        if token.endswith("'s") or token.endswith("’s"):
            return token[:-2]
        return token

    return [remove_suffix(token) for token in tokens]

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


def elasticsearchporter1_tokenizer(text: str) -> List[str]:
    """Recreate Elasticsearch's English analyzer."""
    # Define the regex pattern for Unicode word boundaries
    tokens = standard_tokenizer(text)
    tokens = remove_posessive_suffixes(tokens)
    # Lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in elasticsearch_english_stopwords]
    # Stem tokens
    tokens = [porterv1.stem(token) for token in tokens]
    return tokens


def elasticsearchsnowball_tokenizer(text: str) -> List[str]:
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


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def fold_to_ascii(input_text):
    return unicode_to_ascii(input_text)


def tokenizer(text: str,
              ascii_folding: bool,
              std_tokenizer: bool,
              split_on_case: bool,
              split_on_num: bool,
              remove_possessive: bool,
              stopwords_to_char: Optional[str],
              porter_version: Optional[int]) -> List[str]:
    if ascii_folding:
        text = fold_to_ascii(text)

    if std_tokenizer:
        tokens = standard_tokenizer(text)
    else:
        tokens = text.split()

    # Split on case change FooBar -> Foo Bar
    if split_on_case:
        tokens = unnest_list([split_on_case_change(tok) for tok in tokens])

    # Split on number
    if split_on_num:
        tokens = unnest_list([split_on_char_num_change(tok) for tok in tokens])

    # Lowercase
    tokens = [token.lower() for token in tokens]

    # Strip 's from characters
    if remove_possessive:
        tokens = remove_posessive_suffixes(tokens)

    # Replace stopwords with a character
    if stopwords_to_char:
        tokens = [token if token not in elasticsearch_english_stopwords
                  else stopwords_to_char for token in tokens]

    # Stem with Porter stemmer version if specified
    if porter_version == 1:
        tokens = [porterv1.stem(token) for token in tokens]
    elif porter_version == 2:
        tokens = [porter2_stem_word(token) for token in tokens]

    return tokens


def tokenizer_factory(ascii_folding: bool,
                      std_tokenizer: bool,
                      split_on_case: bool,
                      split_on_num: bool,
                      remove_possessive: bool,
                      stopwords_to_char: Optional[str],
                      porter_version: Optional[int]) -> partial:
    return partial(tokenizer,
                   ascii_folding=ascii_folding,
                   std_tokenizer=std_tokenizer,
                   split_on_case=split_on_case,
                   split_on_num=split_on_num,
                   remove_possessive=remove_possessive,
                   stopwords_to_char=stopwords_to_char,
                   porter_version=porter_version)


def tokenizer_from_str(tok_str):
    """
    Each char corresponds to a different tokenizer setting.
    """
    # Validate args
    if len(tok_str) != 7:
        raise ValueError("Tokenizer string must be 7 characters long")
    if tok_str[0] not in 'aN':
        raise ValueError("First character must be either 'a' (ascii folding) or 'N' (no ascii folding)")
    if tok_str[1] not in 'sw':
        raise ValueError("Second character must be either 's' (standard tokenizer) or 'w' (whitespace tokenizer)")
    if tok_str[2] not in 'cN':
        raise ValueError("Third character must be either 's' (split on case) or 'N' (don't split on case)")
    if tok_str[3] not in 'nN':
        raise ValueError("Third character must be either 'n' (split on number) or 'N' (don't split on case)")
    if tok_str[4] not in 'pN':
        raise ValueError("Third character must be either 'p' (remove possessive) or 'N' (don't remove possessive)")
    if tok_str[5] not in 'sN':
        raise ValueError("Third character must be either 's' (stopwords to char) or 'N' (don't stopwords to char)")
    if tok_str[6] not in '12N':
        raise ValueError("Third character must be either '1' (porter version 1), '2' (porter version 2), or 'N' (no stemming)")
    porter_version = int(tok_str[6]) if tok_str[6] != 'N' else None

    return tokenizer_factory(
        tok_str[0] == 'a',
        std_tokenizer=tok_str[1] == 's',
        split_on_case=tok_str[2] == 'c',
        split_on_num=tok_str[3] == 'n',
        remove_possessive=tok_str[4] == 'p',
        stopwords_to_char='_' if tok_str[5] == 's' else None,
        porter_version=porter_version
    )


def every_tokenizer_str():
    case = 'N'
    num = 'N'
    for ascii_fold in ['a', 'N']:
        for tok in ['s', 'w']:
            for poss in ['p', 'N']:
                for stop in ['s', 'N']:
                    for stem in ['1', '2', 'N']:
                        yield f"{ascii_fold}{tok}{case}{num}{poss}{stop}{stem}"


def every_tokenizer():
    for tok_str in every_tokenizer_str():
        yield tokenizer_from_str(tok_str), tok_str


def test():
    std_tokenizer = tokenizer_from_str("NsNNNNN")
    ws_tokenizer = tokenizer_from_str("NwNNNNN")
    assert std_tokenizer('👍👎') == ['👍', '👎']
    assert ws_tokenizer('👍👎') == ['👍👎']


    ascii_fold = tokenizer_from_str("asNNNNN")
    no_ascii_fold = tokenizer_from_str("NsNNNNN")
    assert ascii_fold("René") == ["rene"]
    assert no_ascii_fold("René") == ["rené"]
    assert (ascii_fold("àáâãäåçèéêëìíîïðñòóôõöøùúûüýþÿ")
            == ['aaaaaaceeeeiiiidnoooooouuuuyty'])

    split_on_case_change = tokenizer_from_str("NscNNNN")
    no_split_on_case_str = tokenizer_from_str("NsNNNNN")
    assert no_split_on_case_str("fooBar") == ["foobar"]
    assert split_on_case_change("fooBar") == ["foo", "bar"]

    porter1 = tokenizer_from_str("NsNNNN1")
    porter2 = tokenizer_from_str("NsNNNN2")
    no_stem = tokenizer_from_str("NsNNNNN")
    assert porter1("1920s") == ["1920"]
    assert porter2("1920s") == ["1920s"]
    assert no_stem("running") == ["running"]

    stopwords = tokenizer_from_str("NsNNNsN")
    no_stopwords = tokenizer_from_str("NsNNNNN")
    assert stopwords("the") == ["_"]
    assert no_stopwords("the") == ["the"]

    posessive = tokenizer_from_str("NsNNpNN")
    no_possessive = tokenizer_from_str("NsNNNNN")
    assert posessive("the's") == ["the"]
    assert no_possessive("the") == ["the"]

    split_on_num = tokenizer_from_str("NsNnNNN")
    no_split_on_sum = tokenizer_from_str("NsNNNNN")
    assert split_on_num("foo2thee") == ["foo", "2", "thee"]
    assert no_split_on_sum("foo2thee") == ["foo2thee"]

    counter = 0
    for tok, tok_str in every_tokenizer():
        tok("René fooBar 1920s running the the's foo2thee 👍👎 FooBar")
        tok(""" Takes a word and a list of suffix-removal rules represented as
        3-tuples, with the first element being the suffix to remove,
        the second element being the string to replace it with, and the
        final element being the condition for the rule to be applicable,
        or None if the rule is unconditional.""")
        tok("""Darkness on the Edge of Town is the fourth studio album by the American singer-songwriter Bruce Springsteen (pictured), released on June 2, 1978, by Columbia Records. The album was recorded during sessions in New York City with the E Street Band from June 1977 to March 1978, after a series of legal disputes between Springsteen and his former manager Mike Appel. Darkness musically strips the Wall of Sound production of its predecessor, Born to Run, for a rawer hard rock sound emphasizing the band as a whole. The lyrics focus on ill-fortuned characters who fight back against overwhelming odds. Released three years after Born to Run, Darkness did not sell as well but reached number five in the United States. Critics initially praised the album's music and performances but were divided on the lyrical content. In recent decades, Darkness has attracted acclaim as one of Springsteen's best works and has appeared on lists of the greatest albums of all time. (Full article...)""")

        counter += 1


if __name__ == "__main__":
    for i in range(100):
        test()
