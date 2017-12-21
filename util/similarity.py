import numpy as np
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from numpy.linalg import norm
from functools import reduce
import re
from fastdtw import fastdtw
import phonetics


def revert_pitches_dementia(pitches):
    # revert ugly hack from before
    pitches = re.sub('(#){2,}', '#', pitches)
    if pitches[0] == '#':
        pitches = pitches[1:]
    if pitches[-1] == '#':
        pitches = pitches[:-1]
    return pitches


def average_pitch_per_token(token_pitch, octave_invariant=False):
    pitches_without_braindamage = list(map(revert_pitches_dementia, token_pitch))
    if octave_invariant:
        return list(map(lambda x: reduce(lambda x, elem: x + int(elem) % 8, x.split('#'), 0) / len(x.split('#')),
                        pitches_without_braindamage))
    else:
        return list(map(lambda x: reduce(lambda x, elem: x + int(elem), x.split('#'), 0) / len(x.split('#')),
                        pitches_without_braindamage))


def string_similarity(some, other):
    return Levenshtein().get_sim_score(some, other)


# hacky and slow
def dtw_string_similarity(some, other):
    if not some or not other:
        return 0.0
    return 1 - dtw_normalized(list(map(hash, list(some))), list(map(hash, list(other))),
                              metric=lambda x, y: 0 if x == y else 1)


def dtw_normalized(x, y, metric=lambda x, y: abs(x - y)):
    return fastdtw(x, y, dist=metric)[0] / np.maximum(len(x), len(y))


def phonetic_similarity(some, other, use_equivalences=False):
    if some == other:
        return 1.0
    if not some or not other:
        return 0.0

    some_phonetics = phonetics.dmetaphone(some)
    other_phonetics = phonetics.dmetaphone(other)
    if some_phonetics == other_phonetics:
        return 1.0

    pair_wise_similarities = []
    for some_phonetic in some_phonetics:
        if not some_phonetic:
            continue
        for other_phonetic in other_phonetics:
            if not other_phonetic:
                continue
            some_equiv = metaphone_representative(some_phonetic) if use_equivalences else some_phonetic
            other_equiv = metaphone_representative(other_phonetic) if use_equivalences else other_phonetic
            pair_wise_similarities.append(string_similarity(some_equiv, other_equiv))
    return 0.0 if not pair_wise_similarities else max(pair_wise_similarities)


# paths: list of t list -> list of t list
# paths [[a,b]] = [[a], [b]]
# paths [[a], [b,c], [d]] = [[a,b,d], [a,c,d]]
def paths(xs):
    if not xs:
        return []
    if len(xs) == 1:
        return [[xs_elem] for xs_elem in xs[0]]
    else:
        p = []
        for prefix in paths([xs[0]]):
            for suffix in paths(xs[1:]):
                p.append(prefix + suffix)
        return p


def string_collate(xs):
    return reduce(lambda x, elem: x + elem, xs, '')


# given a lists of tokens, compute the lists' phonetics' similarity
# compute the phonetics token-wise
def phonetic_similarity_lists(some, other, use_equivalences=False):
    if some == other:
        return 1.0
    if not some or not other:
        return 0.0
    some_phonetics = list(map(phonetics.dmetaphone, some))
    some_phonetics_non_empty = reduce(lambda x, elem: x + [[alt for alt in elem if alt]], some_phonetics, [])
    some_phonetics_paths = paths(some_phonetics_non_empty)
    some_phonetics_paths_collated = list(map(string_collate, some_phonetics_paths))
    # print('ps:', some_phonetics_paths_collated)

    other_phonetics = list(map(phonetics.dmetaphone, other))
    other_phonetics_non_empty = reduce(lambda x, elem: x + [[alt for alt in elem if alt]], other_phonetics, [])
    other_phonetics_paths = paths(other_phonetics_non_empty)
    other_phonetics_paths_collated = list(map(string_collate, other_phonetics_paths))
    # print('po:', other_phonetics_paths_collated)

    if some_phonetics_paths_collated == other_phonetics_paths_collated:
        return 1.0

    pair_wise_similarities = []
    for some_phonetic in some_phonetics_paths_collated:
        if not some_phonetic:
            continue
        for other_phonetic in other_phonetics_paths_collated:
            if not other_phonetic:
                continue
            some_equiv = metaphone_representative(some_phonetic) if use_equivalences else some_phonetic
            other_equiv = metaphone_representative(other_phonetic) if use_equivalences else other_phonetic
            pair_wise_similarities.append(string_similarity(some_equiv, other_equiv))
    return 0.0 if not pair_wise_similarities else max(pair_wise_similarities)


# put all plosives, nasals, fricatives, approximants into each one equivalence class
def metaphone_representative(phonetic):
    phonetic = re.sub('[PTK]', 'P', phonetic)
    phonetic = re.sub('[MN]', 'N', phonetic)
    phonetic = re.sub('[XRSFH]', 'F', phonetic)
    phonetic = re.sub('[LJ]', 'L', phonetic)
    return phonetic


# faster way to compute cosine similarity (skips the input validation)
def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))


# vec_store: vector store for words word_i
def cosine_similarity_with_store(word_u, word_v, vec_store):
    if word_u == word_v:
        return 1.0
    u = vec_store[word_u]
    v = vec_store[word_v]
    return np.dot(u, v) / (norm(u) * norm(v))


# intrinsically normalizes the matrix by its maximum value
def normalize_distance_matrix(matrix):
    max_elem = max(list(map(max, matrix)))
    return list(map(lambda row: list(map(lambda x: 1 - x / max_elem, row)), matrix))


# abstract over pos tags by clustering them
# assumes a Penn Treebank Tags input tag
def tag_to_super_tag(tag, tts_mapping):
    return tts_mapping.get(tag, tag)


def load_tag_to_super_tag_mapping():
    return {'VB': 'V', 'VBG': 'V', 'VBP': 'V', 'VBZ': 'V', 'VBD': 'V', 'VBN': 'V', 'VBD': 'V', 'VBN': 'V',
            'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
            'WDT': 'WH', 'WP': 'WH', 'WP$': 'WH', 'WRB': 'WH',
            '#': '?', '$': '?', '.': '?', ',': '?', ':': '?', '-RRB-': '?', '-LRB-': '?', '“': '?', '”': '?'}

# Tests for paths(.)
# ex1 = [['a','b']]
# ex2 = [['a'], ['b']]
# ex3 = [['a','b'], ['c']]
# ex4 = [['a'], ['b','c']]
# ex5 = [['a','b'], ['c','d']]
# ex6 = [['a','b'], ['c'], ['d','e']]
# ex7 = [['a'], ['b','c'], ['d']]
# print(paths(ex1) == [['a'], ['b']])
# print(paths(ex2) == [['a','b']])
# print(paths(ex3) == [['a','c'], ['b','c']])
# print(paths(ex4) == [['a','b'], ['a','c']])
# print(paths(ex5) == [['a','c'], ['a','d'], ['b','c'], ['b','d']])
# print(paths(ex6) == [['a','c','d'], ['a','c','e'], ['b','c','d'], ['b','c','e']])
# print(paths(ex7) == [['a','b','d'], ['a','c','d']])
# paths(ex6)
