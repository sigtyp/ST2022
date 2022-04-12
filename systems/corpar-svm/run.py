"""
Test List et al.'s SVM-based CorPaR-classifer on the data.
"""
from lingrex.reconstruct import transform_alignment, OneHot
from sklearn.svm import SVC
from functools import partial
from sigtypst2022 import sigtypst2022_path, load_cognate_file, write_cognate_file
from collections import defaultdict
from tqdm import tqdm as progressbar
import argparse


align_f = partial(
        transform_alignment, align=True, position=False, prosody=True,
        startend=True)


class CorPaRSVM(object):
    """
    SVM classifier for correspondence patterns.
    
    :param datapath: The path to the input data.
    :param gap: Gap symbol used for alignments.
    :param missing: Missing data symbol.      
    """

    def __init__(
            self, datapath, gap="-", missing="Ø"):
        self.languages, self.sounds, self.data = load_cognate_file(datapath)
        self.gap, self.missing = gap, missing

        # make a simple numerical embedding for sounds
        self.classifiers = {
            language: SVC(kernel="linear") for language in self.languages}
        self.onehots = {}
        self.alignments = {
                language: [] for language in self.languages} 

        for cogid, data in self.data.items():
            alms, languages = [], []
            for language in self.languages:
                if data[language] and \
                        " ".join(data[language]) != "?":
                    alms += [data[language]]
                    languages += [language]
            for i, language in enumerate(languages):
                self.alignments[language].append(
                        [
                            cogid,
                            [lang for lang in languages if lang != language]+[language],
                            [alm for j, alm in enumerate(alms) if i != j]+[alms[i]]
                            ]
                        )


    def fit(self, func=align_f):
        """
        Fit the data.

        :param func: The alignment function to be used to align the data.
        """
        print("[i] fit the clf")
        self.patterns = defaultdict(lambda : defaultdict(list))
        self.func = func
        self.matrices = {language: [] for language in self.languages}
        self.solutions = {language: [] for language in self.languages}
        self.patterns = {
                language: defaultdict(lambda : defaultdict(list)) for
                language in self.languages}
        self.sounds2idxs = {language: {self.gap: 1, self.missing: 0} for
                language in self.languages}
        self.tsounds2idxs = {language: {self.gap: 1, self.missing: 0} for
                language in self.languages}
        self.idxs2sounds = {language: {} for language in self.languages}
        self.idxs2tsounds = {language: {} for language in self.languages}
        for language in progressbar(self.languages, desc="aligning data"):
            sounds, tsounds = defaultdict(int), defaultdict(int)
            for cogid, languages, alms in self.alignments[language]:
                alm_matrix = self.func(
                        alms, languages, 
                        [l for l in self.languages if l != language]+[language], 
                        training=True)
                for i, row in enumerate(alm_matrix):
                    ptn = tuple(row[:len(self.languages)-1]+row[len(self.languages):])
                    self.patterns[language][ptn][row[len(self.languages)-1]] += [(cogid, i)]
                    for sound in ptn:
                        sounds[sound] += 1
                    tsounds[row[len(self.languages)-1]] += 1
            for i, sound in enumerate(sorted(sounds, key=lambda x: sounds[x], reverse=True)):
                self.sounds2idxs[language][sound] = i+2
            for i, sound in enumerate(sorted(tsounds, key=lambda x: tsounds[x], reverse=True)):
                self.tsounds2idxs[language][sound] = i+2
            self.idxs2sounds[language] = {v: k for k, v in
                    self.sounds2idxs[language].items()}
            self.idxs2tsounds[language] = {v: k for k, v in
                    self.tsounds2idxs[language].items()}

        for language in progressbar(self.languages, desc="fitting classifiers"):
            for pattern, sounds in self.patterns[language].items():
                for sound, vals in sounds.items():
                    target = self.tsounds2idxs[language][sound]
                    row = [self.sounds2idxs[language][s] for s in pattern]
                    for cogid, idx in vals:
                        self.matrices[language] += [row]
                        self.solutions[language] += [target]
            self.onehots[language] = OneHot(self.matrices[language])

            self.classifiers[language].fit(
                    self.onehots[language](self.matrices[language]),
                    self.solutions[language])
    
    def predict(self, languages, words, target, unknown="?"):
        """
        Predict a word for a given language.

        :param languages: The list of languages corresponding to the sequences.
        :param words: The list of words from which to predict.
        :param target: The name of the target language into which to predict.
        :param unknown: The symbol to used for unknown predictions.
        """
        
        matrix = self.func(
                words, languages, [l for l in self.languages if l !=
                    target],
                training=False,
                )

        new_matrix = [[0 for char in row] for row in matrix]
        for i, row in enumerate(matrix):
            for j, char in enumerate(row):
                new_matrix[i][j] = self.sounds2idxs[target].get(char, 0)
        oh_matrix = self.onehots[target](new_matrix)
        out = [self.idxs2tsounds[target].get(idx, unknown) for idx in
                self.classifiers[target].predict(oh_matrix)]

        return [x for x in out if x != "-"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo of SVM CorPaR Method')
    parser.add_argument(
            "--surprise", 
            action="store_true",
            help="Analyze surprise data."
            )

    args = parser.parse_args()
    
    datapath, outpath = "data", "training"
    if args.surprise:
        datapath = "data-surprise"
        outpath = "surprise"


    # iterate over the different proportions in the data
    for prop in ["0.10", "0.20", "0.30", "0.40", "0.50"]:
        print("[i] analyzing proportion {0}".format(prop))

        # use sigtyst2022_path to get the data
        training = sigtypst2022_path(datapath).glob("*/training-{0}.tsv".format(prop))
        predict = sigtypst2022_path(datapath).glob("*/test-{0}.tsv".format(prop))

        # iterate over the files
        for f1, f2 in zip(training, predict):
            ds = str(f1).split("/")[-2]
            print("[i] analyzing dataset {0}".format(ds))

            # fit the classifiers
            clf = CorPaRSVM(f1)
            clf.fit()

            # load the test data
            languages, sounds, testdata = load_cognate_file(f2)
            predictions = defaultdict(dict)

            # iterate over cognate sets and prepare the words from which to
            # predict
            for cogid, values in progressbar(testdata.items(), desc="predicting words"):
                alms, current_languages = [], []
                target = ""
                for language in languages:
                    if language in values and " ".join(values[language]) not in ["?", ""]:
                        alms += [values[language]]
                        current_languages += [language]
                    elif " ".join(values[language]) == "?":
                        target = language

                if alms and target:
                    out = clf.predict(current_languages, alms, target)
                    predictions[cogid][target] = out
            
            # write data to file
            write_cognate_file(
                    clf.languages, predictions,  
                    sigtypst2022_path("systems", "corpar-svm", outpath, ds, "result-{0}.tsv".format(prop)))

