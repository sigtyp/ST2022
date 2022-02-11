"""Utility functions and data handling for the shared task."""

from lingpy import *
from lingpy.evaluate.acd import _get_bcubed_score as bcubed_score
from pathlib import Path
from git import Repo
from lingpy.compare.partial import Partial
import argparse
from collections import defaultdict
import random
import networkx as nx
from networkx.algorithms.clique import find_cliques
from lingpy.align.sca import get_consensus
from lingpy.sequence.sound_classes import prosodic_string, class2tokens
from lingpy.align.multiple import Multiple
from itertools import combinations
from tabulate import tabulate
import json


def download(datasets, pth):
    """
    Download all datasets as indicated with GIT.
    """
    for dataset, conditions in datasets.items():
        if pth.joinpath(dataset, "cldf", "cldf-metadata.json").exists():
            print("skipping existing dataste {0}".format(dataset))
        else:
            repo = Repo.clone_from(
                    "https://github.com/"+conditions["path"]+".git",
                    pth / dataset)
            repo.git.checkout(conditions["version"])
            print("downloaded {0}".format(dataset))


def prepare(datasets, datapath, cldfdatapath, runs=1000):
    """
    Function computes cognates from a CLDF dataset and writes them to file.
    """
    for dataset, conditions in datasets.items():
        print("[i] analyzing {0}".format(dataset))
        columns = [
            "parameter_id",
            "concept_name",
            "language_id",
            "language_name",
            "value",
            "form",
            "segments",
            "language_glottocode",
            "language_latitude",
            "language_longitude",
            "language_"+conditions["subgroup"]
            ]

        if conditions["cognates"] == "true":
            columns += ["cogid_cognateset_id"]
        # preprocessing to get the subset of the data
        wl = Wordlist.from_cldf(
            cldfdatapath.joinpath(dataset, "cldf", "cldf-metadata.json"),
            columns=columns
            )
        D = {0: [h for h in wl.columns]}
        for idx, subgroup in wl.iter_rows("language_"+conditions["subgroup"]):
            if subgroup == conditions["name"]:
                D[idx] = wl[idx]
        if conditions["cognates"] == "false":
            part = Partial(D)
            part.get_partial_scorer(runs=runs)
            part.partial_cluster(method="lexstat", threshold=0.45, ref="cogids",
                    cluster_method="infomap")
            # check for lingrex here as well
            etd = part.get_etymdict(ref="cogids")
            cognates = {}
            for cogid, idxs_ in etd.items():
                idxs, count = {}, 0
                for idx, language in zip(idxs_, part.cols):
                    if idx:
                        tks = part[idx[0], "tokens"]
                        cogidx = part[idx[0], "cogids"].index(cogid)
                        idxs[language] = " ".join([
                            x.split("/")[1] if "/" in x else x for x in
                            tks.n[cogidx]])
                        count += 1
                    else:
                        idxs[language] = ""
                if count >= 2:
                    cognates[cogid] = idxs
        else:
            part = Wordlist(D)
            etd = part.get_etymdict(ref="cogid")
            cognates = {}
            for cogid, idxs_ in etd.items():
                idxs, count = {}, 0
                for idx, language in zip(idxs_, part.cols):
                    if idx:
                        tks = part[idx[0], "tokens"]
                        idxs[language] = " ".join([x.split("/")[1] if "/" in x
                            else x for x in tks])
                        count += 1
                    else:
                        idxs[language] = ""
                if count >= 2:
                    cognates[cogid] = idxs
            

        if datapath.joinpath(dataset).exists():
            pass
        else:
            Path.mkdir(datapath.joinpath(dataset))
        with open(datapath.joinpath(dataset, "cognates.tsv"), "w") as f:
            f.write("COGID\t"+"\t".join(part.cols)+"\n")
            for cogid, idxs in cognates.items():
                f.write("{0}".format(cogid))
                for language in part.cols:
                    f.write("\t{0}".format(idxs[language]))
                f.write("\n")
        wl.output(
                "tsv", filename=datapath.joinpath(dataset, "wordlist").as_posix(), ignore="all", prettify=False)


def load_cognate_file(path):
    """
    Helper function for simplified cognate formats.
    """
    data = csv2list(path, strip_lines=False)
    header = data[0]
    languages = header[1:]
    out = {}
    sounds = defaultdict(lambda : defaultdict(list))
    for row in data[1:]:
        out[row[0]] = {}
        for language, entry in zip(languages, row[1:]):
            out[row[0]][language] = entry.split()
            for i, sound in enumerate(entry.split()):
                sounds[sound][language] += [[row[0], i]]
    return languages, sounds, out



def write_cognate_file(languages, data, path):
    with open(path, "w") as f:
        f.write("COGID\t"+"\t".join(languages)+"\n")
        for k, v in data.items():
            f.write("{0}".format(k))
            for language in languages:
                f.write("\t"+" ".join(v.get(language, [])))
            f.write("\n")



def split_training_test_data(data, languages, ratio=0.1):
    """
    Split data into test and training data.
    """
    split_off = int(len(data) * ratio + 0.5)
    cognates = [key for key, value in sorted(
        data.items(),
        key=lambda x: sum([1 if " ".join(b) not in ["", "?"] else 0 for a, b in
            x[1].items()]),
        reverse=True)
        ]
    test_, training = (
            {c: data[c] for c in cognates[:split_off]}, 
            {c: data[c] for c in cognates[split_off:]}
            )
    
    # now, create new items for all languages to be predicted
    test = defaultdict(dict)
    solutions = defaultdict(dict)
    for i, language in enumerate(languages):
        for key, values in test_.items():
            if " ".join(test_[key][language]):
                new_key = key+"-"+str(i+1)
                for j, languageB in enumerate(languages):
                    if language != languageB:
                        test[new_key][languageB] = test_[key][languageB]
                    else:
                        test[new_key][language] = ["?"]
                        solutions[new_key][language] = test_[key][language]
    
    return training, test, solutions
    

def _split_training(data, ratio=0.1):
    """
    Split data into parts for training and development.
    """
    # determine the number of words per language
    counts = defaultdict(list)
    for key, value in data.items():
        for k, v in value.items():
            if v:
                counts[k] += [key]
    solutions = defaultdict(dict)
    new_data = defaultdict(dict)
    for language, keys in counts.items():
        cut = int(ratio * len(keys)+0.5)
        sampled = random.sample(keys, cut)
        for key in [k for k in keys if k not in sampled]:
            new_data[key][language] = data[key][language]
        for key in sampled:
            solutions[key][language] = data[key][language]
            new_data[key][language] = ["?"]
    return new_data, solutions
        

def split_data(datasets, pth, props=None):
    props = props or [0.1, 0.2, 0.3, 0.4, 0.5]

    for prop in props:
        for dataset, conditions in datasets.items():
            languages, sounds, data = load_cognate_file(
                    pth.joinpath(dataset, "cognates.tsv"))
            #data_part, solutions = split_training(data, ratio=prop)
            training, test, solutions = split_training_test_data(
                    data, languages, ratio=prop)
            write_cognate_file(
                    languages, 
                    training,
                    pth.joinpath(
                        dataset, "training-{0:.2f}.tsv".format(prop)),
                    )
            write_cognate_file(
                    languages, 
                    test,
                    pth.joinpath(
                        dataset, "test-{0:.2f}.tsv".format(prop)),
                    )
            write_cognate_file(
                    languages,
                    solutions,
                    pth.joinpath(
                        dataset, "solutions-{0:.2f}.tsv".format(prop)),
                    )
            print("wrote training and solution data for {0} / {1:.2f}".format(
                dataset, prop))



def ungap(alignment, languages, proto):
    cols = []
    pidxs = []
    for i, taxon in enumerate(languages):
        if taxon == proto:
            pidxs += [i]
    merges = []
    for i in range(len(alignment[0])):
        col = [row[i] for row in alignment]
        col_rest = [site for j, site in enumerate(col) if j not in pidxs]
        if "-" in col_rest and len(set(col_rest)) == 1:
            merges += [i]
    if merges:
        new_alms = []
        for i, row in enumerate(alignment):
            new_alm = []
            mergeit = False
            started = True
            for j, cell in enumerate(row):
                if j in merges or mergeit:
                    mergeit = False
                    if not started: #j != 0:
                        if cell == "-":
                            pass
                        else:
                            if not new_alm[-1]:
                                new_alm[-1] += cell
                            else:
                                new_alm[-1] += '.'+cell
                    else:
                        mergeit = True
                        if cell == "-":
                            new_alm += [""]
                        else:
                            new_alm += [cell]
                else:
                    started = False
                    new_alm += [cell]
            for k, cell in enumerate(new_alm):
                if not cell:
                    new_alm[k] = "-"
            new_alms += [new_alm]
        return new_alms
    return alignment


class CorPaRClassifier(object):

    def __init__(self, minrefs=2, missing=0, threshold=1):
        """
        Word prediction method adopted from List (2019).
        """
        self.G = nx.Graph()
        self.missing = 0
        self.threshold = threshold

    def compatible(self, ptA, ptB):
        match_, mismatch = 0, 0
        for a, b in zip(ptA, ptB):
            if not a or not b:
                pass
            elif a == b:
                match_ += 1
            else:
                mismatch += 1
        return match_, mismatch

    def consensus(self, nodes):
        
        cons = []
        for i in range(len(nodes[0])):
            nocons = True
            for node in nodes:
                if node[i] != self.missing:
                    cons += [node[i]]
                    nocons = False
                    break
            if nocons:
                cons += [self.missing]
        return tuple(cons)

    def fit(self, X, y):
        """
        Fit the data to the classifier.
        """
        # get identical patterns
        P = defaultdict(list)
        for i, row in enumerate(X):
            P[tuple(row+[y[i]])] += [i]
        # make graph
        for (pA, vA), (pB, vB) in combinations(P.items(), r=2):
            match, mismatch = self.compatible(pA, pB)
            if not mismatch and match >= self.threshold:
                if not pA in self.G:
                    self.G.add_node(pA, freq=len(vA))
                if not pB in self.G:
                    self.G.add_node(pB, freq=len(vB))
                self.G.add_edge(pA, pB, weight=match)
        self.patterns = defaultdict(lambda : defaultdict(list))
        self.lookup = defaultdict(lambda : defaultdict(int))
        # get cliques
        for nodes in find_cliques(self.G):
            cons = self.consensus(list(nodes))
            self.patterns[cons[:-1]][cons[-1]] = len(nodes)
            for node in nodes:
                self.lookup[node[:-1]][cons[:-1]] += len(nodes)
        self.candidates = {}
        self.predictions = {}
        for ptn in self.patterns:
            self.predictions[ptn] = [x for x, y in sorted(
                self.patterns[ptn].items(),
                key=lambda p: p[1],
                reverse=True)][0]
        for ptn in self.lookup:
            ptnB = [x for x, y in sorted(self.lookup[ptn].items(),
                key=lambda p: p[1],
                reverse=True)][0]
            self.predictions[ptn] = self.predictions[ptnB]

        # make index of data points for quick search based on attested data
        self.ptnlkp = defaultdict(list)
        for ptn in self.patterns:
            for i in range(len(ptn)):
                if ptn[i] != self.missing:
                    self.ptnlkp[i, ptn[i]] += [ptn]

    def predict(self, matrix):
        out = []
        for row in matrix:
            ptn = tuple(row)
            try:
                out += [self.predictions[ptn]]
            except KeyError:
                candidates = []
                visited = set()
                for i in range(len(ptn)-1):
                    if ptn[i] != self.missing:
                        for ptnB in self.ptnlkp[i, ptn[i]]:
                            if ptnB not in visited:
                                visited.add(ptnB)
                                match, mismatch = self.compatible(ptn, ptnB)
                                if match and not mismatch:
                                    candidates += [(ptnB, match+len(ptn))]
                                elif match-mismatch:
                                    candidates += [(ptnB, match-mismatch)]
                if candidates:
                    ptn = [x for x, y in sorted(
                        candidates,
                        key=lambda p: p[1],
                        reverse=True)][0]
                    self.predictions[tuple(row)] = self.predictions[ptn]
                    out += [self.predictions[tuple(row)]]
                else:
                    out += [self.missing]
        return out


def simple_align(
        seqs, 
        languages, 
        all_languages,
        align=True,
        training=True,
        missing="Ø", 
        gap="-",
        ):
    """
    Simple alignment function that inserts entries for missing data.
    """
    if align:
        seqs = [[s for s in seq if s != gap] for seq in seqs]
        msa = Multiple([[s for s in seq if s != gap] for seq in seqs])
        msa.prog_align()
        alms = [alm for alm in msa.alm_matrix]
    else:
        seqs = [[s for s in seq if s != gap] for seq in seqs]
        alms = normalize_alignment([s for s in seqs])
    if training:
        alms = ungap(alms, languages, languages[-1])
        these_seqs = seqs[:-1]
    else:
        these_seqs = seqs
    matrix = [[missing for x in all_languages] for y in alms[0]]
    for i in range(len(alms[0])):
        for j, lng in enumerate(languages):
            lidx = all_languages.index(lng)
            matrix[i][lidx] = alms[j][i]    
    # for debugging
    for row in matrix:
        assert len(row) == len(matrix[0])
    return matrix


class Baseline(object):

    def __init__(
            self, datapath, minrefs=2, missing="Ø", gap="-", threshold=1,
            func=simple_align):
        """
        The baseline is the prediction method by List (2019).
        """
        self.languages, self.sounds, self.data = load_cognate_file(datapath)
        self.gap, self.missing = gap, missing

        # make a simple numerical embedding for sounds
        self.classifiers = {
            language: CorPaRClassifier(minrefs, missing=0,
                    threshold=threshold) for language in self.languages}
        self.alignments = {
                language: [] for language in self.languages} 
        self.to_predict = defaultdict(list)

        for cogid, data in self.data.items():
            alms, languages = [], []
            for language in self.languages:
                if data[language] and \
                        " ".join(data[language]) != "?":
                    alms += [data[language]]
                    languages += [language]
                elif data[language] and " ".join(data[language]) == "?":
                    self.to_predict[cogid] += [language]
            for i, language in enumerate(languages):
                self.alignments[language].append(
                        [
                            cogid,
                            [lang for lang in languages if lang != language]+[language],
                            [alm for j, alm in enumerate(alms) if i != j]+[alms[i]]
                            ]
                        )
        self.func = func


    def fit(self, func=simple_align):
        """
        Fit the data.
        """
        self.patterns = defaultdict(lambda : defaultdict(list))
        self.func = func
        self.matrices = {language: [] for language in self.languages}
        self.solutions = {language: [] for language in self.languages}
        self.patterns = {
                language: defaultdict(lambda : defaultdict(list)) for
                language in self.languages}
        sounds = set()
        for language in self.languages:
            for cogid, languages, alms in self.alignments[language]:
                alm_matrix = self.func(
                        alms, languages, self.languages,
                        training=True)
                for i, row in enumerate(alm_matrix):
                    ptn = tuple(row[:len(self.languages)-1])
                    self.patterns[language][ptn][row[-1]] += [(cogid, i)]
                    for sound in ptn:
                        sounds.add(sound)
                    sounds.add(row[-1])
        self.sound2idx = dict(zip(sorted(sounds), range(2, len(sounds)+2)))
        self.sound2idx[self.gap] = 1
        self.sound2idx[self.missing] = 0
        self.idx2sound = {v: k for k, v in self.sound2idx.items()}

        for language in self.languages:
            for pattern, sounds in self.patterns[language].items():
                for sound, vals in sounds.items():
                    target = self.sound2idx[sound]
                    row = [self.sound2idx[s] for s in pattern]
                    for cogid, idx in vals:
                        self.matrices[language] += [row]
                        self.solutions[language] += [target]
            print("fitting classified for {0}".format(language))
            self.classifiers[language].fit(
                    self.matrices[language],
                    self.solutions[language])
            print('... fitted the classifier')
    
    def predict(self, languages, alignments, target, unknown="?"):
        
        matrix = self.func(
                alignments, languages, [l for l in self.languages if l !=
                    target],
                training=False,
                )
        new_matrix = [[0 for char in row] for row in matrix]
        for i, row in enumerate(matrix):
            for j, char in enumerate(row):
                new_matrix[i][j] = self.sound2idx.get(char, 0)
        out = [self.idx2sound.get(idx, unknown) for idx in
                self.classifiers[target].predict(new_matrix)]
        return [x for x in out if x != "-"]


def predict_words(ifile, pfile, ofile):

    bs = Baseline(ifile)
    bs.fit()
    languages, sounds, testdata = load_cognate_file(pfile)
    predictions = defaultdict(dict)
    for cogid, values in testdata.items():
        alms, current_languages = [], []
        target = ""
        for language in languages:
            if language in values and " ".join(values[language]) not in ["?", ""]:
                alms += [values[language]]
                current_languages += [language]
            elif " ".join(values[language]) == "?":
                target = language

            if alms and target:
                out = bs.predict(current_languages, alms, target)
                predictions[cogid][target] = out
    write_cognate_file(bs.languages, predictions, ofile)


def compare_words(firstfile, secondfile, report=True):
    """
    Evaluate the predicted and attested words in two datasets.
    """

    (languages, soundsA, first), (languagesB, soundsB, last) = load_cognate_file(firstfile), load_cognate_file(secondfile)
    all_scores = []
    for language in languages:
        scores = []
        almsA, almsB = [], []
        for key in first:
            if language in first[key]:
                entryA = first[key][language]
                if " ".join(entryA):
                    entryB = last[key][language]
                    almA, almB, _ = nw_align(entryA, entryB)
                    almsA += almA
                    almsB += almB
                    score = 0
                    for a, b in zip(almA, almB):
                        if a == b and a not in "Ø?-":
                            pass
                        elif a != b:
                            score += 1
                    scoreD = score / len(almA)
                    scores += [[key, entryA, entryB, score, scoreD]]
        if scores:
            p, r = bcubed_score(almsA, almsB), bcubed_score(almsB, almsA)
            fs = 2 * (p*r) / (p+r)
            all_scores += [[
                language,
                sum([row[-2] for row in scores])/len(scores),
                sum([row[-1] for row in scores])/len(scores),
                fs]]
    all_scores += [[
        "TOTAL", 
        sum([row[-3] for row in all_scores])/len(languages),
        sum([row[-2] for row in all_scores])/len(languages),
        sum([row[-1] for row in all_scores])/len(languages)
        ]]
    if report:
        print(tabulate(all_scores, headers=["Language", "ED", 
            "ED (Normalized)", "B-Cubed FS"], floatfmt=".3f"))
    return all_scores
    

def main(*args):

    parser = argparse.ArgumentParser(description='ST 2022')
    parser.add_argument(
            "--download", 
            action="store_true"
            )
    parser.add_argument(
            "--datapath",
            default=Path("data"),
            type=Path,
            action="store"
            )
    parser.add_argument(
            "--cldf-data",
            default=Path("cldf-data"),
            type=Path,
            action="store"
            )
    parser.add_argument(
            "--prepare",
            action="store_true"
            )
    parser.add_argument(
            "--split",
            action="store_true"
            )
    parser.add_argument(
            "--runs",
            action="store",
            type=int,
            default=1000
            )
    parser.add_argument(
            "--seed",
            action="store_true",
            )
    parser.add_argument(
            "--predict",
            action="store_true")
    parser.add_argument(
            "--infile",
            action="store",
            type=Path,
            )
    parser.add_argument(
            "--outfile",
            action="store",
            default=""
            )
    parser.add_argument(
            "--testfile",
            action="store",
            default="",
            help="file containing the test data"
            )
    parser.add_argument(
            "--prediction-file",
            action="store",
            default="",
            help="file storing the predictions"
            )
    parser.add_argument(
            "--solution-file",
            action="store",
            default="",
            help="file storing the solutions for a test"
            )
    parser.add_argument(
            "--compare",
            action="store_true"
            )

    parser.add_argument(
            "--datasets",
            action="store",
            default="datasets.json"
            )

    parser.add_argument(
            "--all",
            action="store_true",
            )

    parser.add_argument(
            "--evaluate",
            action="store_true",
            )

    args = parser.parse_args(*args)
    if args.seed:
        random.seed(1234)
    
    with open(args.datasets) as f:
        DATASETS = json.load(f)


    if args.download:
        download(DATASETS, args.cldf_data)
    
    if args.prepare:
        prepare(DATASETS, args.datapath, args.cldf_data, args.runs)
    
    if args.split:
        split_data(DATASETS, args.datapath, props=None)


    if args.predict:
        if not args.all:
            if not args.outfile:
                args.outfile = Path(str(args.infile)[:-4]+"-out.tsv")
            predict_words(args.infile, args.testfile, args.outfile)
        elif args.all:
            for data, conditions in DATASETS.items():
                predict_words(
                        args.datapath.joinpath(data, "training-0.10.tsv"),
                        args.datapath.joinpath(data, "test-0.10.tsv"),
                        args.datapath.joinpath(data, "result-0.10.tsv")
                        )
    if args.evaluate:
        if args.all:
            results = []
            for data, conditions in DATASETS.items():
                results += [compare_words(
                        args.datapath.joinpath(data, "result-0.20.tsv"),
                        args.datapath.joinpath(data,
                            "solutions-0.20.tsv"),
                        report=False)[-1]]
                results[-1][0] = data
            print(tabulate(results, headers=[
                "DATASET", "ED", "ED (NORM)", "B-CUBED FS"]))


    if args.compare:
        compare_words(args.prediction_file, args.solution_file)

