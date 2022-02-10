"""Utility functions and data handling for the shared task."""
from lingpy import *
from pathlib import Path
from git import Repo
from lingpy.compare.partial import Partial
import argparse



def download(datasets, datapath):
    pth = Path(datapath)
    for dataset, conditions in datasets.items():
        if pth.joinpath(dataset, "cldf", "cldf-metadata.json").exists():
            print("skipping existing dataste {0}".format(dataset))
        else:
            repo = Repo.clone_from(
                    "https://github.com/"+conditions["path"]+".git",
                    Path(datapath) / dataset)
            repo.git.checkout(conditions["version"])
            print("downloaded {0}".format(dataset))


def prepare(datasets, datapath):
    pth = Path(datapath)
    for dataset, conditions in datasets.items():
        # preprocessing to get the subset of the data
        wl = Wordlist.from_cldf(
            pth.joinpath(dataset, "cldf", "cldf-metadata.json"),
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
            )
        D = {0: [h for h in wl.columns]}
        for idx, subgroup in wl.iter_rows("language_"+conditions["subgroup"]):
            if subgroup == conditions["name"]:
                D[idx] = wl[idx]
        part = Partial(D)
        part.get_partial_scorer(runs=1000)
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
            if count >= part.width / 2:
                cognates[cogid] = idxs
        with open(pth.joinpath(dataset, "cognates.tsv"), "w") as f:
            f.write("COGID\t"+"\t".join(part.cols)+"\n")
            for cogid, idxs in cognates.items():
                f.write("{0}".format(cogid))
                for language in part.cols:
                    f.write("\t{0}".format(idxs[language]))
                f.write("\n")
        wl.output(
                "tsv", filename=pth.joinpath(dataset, "wordlist").as_posix(), ignore="all", prettify=False)


def load_cognate_file(path):
    """
    Helper function for simplified cognate formats.
    """
    # get languages, etc.
    pass


def split_training(dataset, datapath, ratio=0.1):
    pass


def baseline(path):
    """
    Apply baseline method to the data.
    """

    # determine number of languages
    # for each language
    pass



def main(*args):

    parser = argparse.ArgumentParser(description='ST 2022')
    parser.add_argument(
            "--download", 
            action="store_true"
            )
    parser.add_argument(
            "--datapath",
            default="data",
            action="store"
            )
    parser.add_argument(
            "--prepare",
            action="store_true"
            )


    args = parser.parse_args(*args)
    
    DATASETS = {
            "abrahammonpa": {
                "subgroup": "subgroup", 
                "name": "Tshanglic",
                "path": "lexibank/abrahammonpa",
                "version": "v3.0",
                },
            "allenbai": {
                "subgroup": "subgroup", "name": "Bai",
                "path": "lexibank/allenbai",
                "version": "v4.0"
                },
            "backstromnorthernpakistan": {
                "subgroup": "family", "name": "Sino-Tibetan",
                "path": "lexibank/backstromnorthernpakistan",
                "version": "v1.0"
                },
    }


    if args.download:
        download(DATASETS, args.datapath)
    if args.prepare:
        prepare(DATASETS, args.datapath)
