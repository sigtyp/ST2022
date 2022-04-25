# NOTE: this is a local copy, with some changes (moslty cosmetic in nature)
# to the sigtyp challenge source file, copied here to make its usage
# easier with our project structure

import lingpy
import math
from collections import defaultdict
from tabulate import tabulate
from lingpy.evaluate.acd import _get_bcubed_score as bcubed_score
from lingpy.sequence.ngrams import get_n_ngrams


def load_cognate_file(path):
    """
    Helper function for simplified cognate formats.
    """
    data = lingpy.csv2list(path, strip_lines=False)
    header = data[0]
    languages = header[1:]
    out = {}
    sounds = defaultdict(lambda: defaultdict(list))
    for row in data[1:]:
        out[row[0]] = {}
        for language, entry in zip(languages, row[1:]):
            out[row[0]][language] = entry.split()
            for i, sound in enumerate(entry.split()):
                sounds[sound][language] += [[row[0], i]]
    return languages, sounds, out


def bleu_score(word, reference, n=4, weights=None, trim=False):
    """
    Compute the BLEU score for predicted word and reference.
    """

    if not weights:
        weights = [1 / n for x in range(n)]

    scores = []
    for i in range(1, n + 1):

        new_wrd = list(get_n_ngrams(word, i))
        new_ref = list(get_n_ngrams(reference, i))
        if trim and i > 1:
            new_wrd = new_wrd[i - 1 : -(i - 1)]
            new_ref = new_ref[i - 1 : -(i - 1)]

        clipped, divide = [], []
        for itm in set(new_wrd):
            clipped += [new_ref.count(itm)]
            divide += [new_wrd.count(itm)]
        scores += [sum(clipped) / sum(divide)]

    # calculate arithmetic mean
    out_score = 1
    for weight, score in zip(weights, scores):
        out_score = out_score * (score**weight)

    if len(word) > len(reference):
        bp = 1
    else:
        bp = math.e ** (1 - (len(reference) / len(word)))
    return bp * (out_score ** (1 / sum(weights)))


def compare_words(firstfile, secondfile, report=True):
    """
    Evaluate the predicted and attested words in two datasets.
    """

    (languages, soundsA, first), (languagesB, soundsB, last) = load_cognate_file(
        firstfile
    ), load_cognate_file(secondfile)
    all_scores = []
    for language in languages:
        scores = []
        almsA, almsB = [], []
        for key in first:
            if language in first[key]:
                entryA = first[key][language]
                if " ".join(entryA):
                    entryB = last[key][language]
                    almA, almB, _ = lingpy.nw_align(entryA, entryB)
                    almsA += almA
                    almsB += almB
                    score = 0
                    for a, b in zip(almA, almB):
                        if a == b and a not in "Ø?-":
                            pass
                        elif a != b:
                            score += 1
                    scoreD = score / len(almA)
                    bleu = bleu_score(entryA, entryB, n=4, trim=False)
                    scores += [[key, entryA, entryB, score, scoreD, bleu]]
        if scores:
            p, r = bcubed_score(almsA, almsB), bcubed_score(almsB, almsA)
            fs = 2 * (p * r) / (p + r)
            all_scores += [
                [
                    language,
                    sum([row[-3] for row in scores]) / len(scores),
                    sum([row[-2] for row in scores]) / len(scores),
                    fs,
                    sum([row[-1] for row in scores]) / len(scores),
                ]
            ]
    all_scores += [
        [
            "TOTAL",
            sum([row[-4] for row in all_scores]) / len(languages),
            sum([row[-3] for row in all_scores]) / len(languages),
            sum([row[-2] for row in all_scores]) / len(languages),
            sum([row[-1] for row in all_scores]) / len(languages),
        ]
    ]
    if report:
        print(
            tabulate(
                all_scores,
                headers=["Language", "ED", "ED (Normalized)", "B-Cubed FS", "BLEU"],
                floatfmt=".3f",
            )
        )
    return all_scores
