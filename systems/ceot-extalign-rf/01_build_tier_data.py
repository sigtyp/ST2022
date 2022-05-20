#!/usr/bin/env python3

"""
Prepare multitier data from sigtyp/ST2022 release.
"""

# Import Python standard libraries
from collections import defaultdict
from pathlib import Path
from typing import *
import argparse
import csv
import glob
import logging

# Import 3rd-party libraries
import lingpy
import pandas as pd

# Import other modules
import multitiers

# todo: allow to set/explore different alignment parameters for lingpy
# todo: port the community detection form the private mt repo?
# todo: add overwrite

BASE_PATH = Path(__file__).parent


def train2mt(filename: str) -> Path:
    """
    Builds an mt filename from a corresponding train filename.
    """

    # todo: rewrite with pathlib
    dataset, trainfile = filename.split("/")[-2:]

    return BASE_PATH / "mtdata" / dataset / trainfile


def mt2df(filename: str) -> Path:
    """
    Builds a df filename from a corresponding mt filename.
    """

    # todo: rewrite with pathlib
    mtfile = filename.replace("mtdata", "dfs")

    return Path(mtfile)


def extend_with_alignment(data: List[dict]) -> List[dict]:
    """
    Extends data as provided with alignments.
    """

    # Align for each cogid/line
    mt_data = []
    for line in data:
        # extract cogid
        cogid = line.pop("COGID")

        # collect a map of forms to aligned forms (remember they might
        # be repeated) so that we can collect a language to aligned
        # form dictionary; we use sorted because we need an iterable
        # anyway, and debugging is easier. Note that we skip over forms
        # reported as "?" (entries deleted for testing)
        forms = sorted(
            set([value for value in line.values() if value and value != "?"])
        )
        msa = lingpy.Multiple(forms)
        msa.prog_align(model="asjp")
        alm_forms = {src: tgt for src, tgt in zip(forms, msa.alm_matrix)}

        # extend data to the format expected by the multitiers library
        for lect, form in line.items():
            if form and form != "?":
                mt_data.append(
                    {
                        "DOCULECT": lect,
                        "TOKENS": form,
                        "ALIGNMENT": " ".join(alm_forms[form]),
                        "COGID": cogid,
                    }
                )

    # Sort data, add IDs in-place, and return; we need to perform an operation on the COGID,
    # for sorting, due to the suffix added to test data
    def _mock_cogid(cogid_label):
        # keep track of negative cogids
        negative = False
        if cogid_label[0] == "-":
            negative = True
            cogid_label = cogid_label[1:]

        # drop any suffix
        cogid_label = cogid_label.replace("-", "")

        # add back negative if necessary
        if negative:
            cogid_label = f"-{cogid_label}"

        # return as an integer for comparison
        return int(cogid_label)

    mt_data = sorted(mt_data, key=lambda e: (_mock_cogid(e["COGID"]), e["DOCULECT"]))
    for idx, entry in enumerate(mt_data):
        entry["ID"] = str(idx + 1)

    return mt_data


def prepare_data_test(test_file: str, solution_file: str) -> List[dict]:
    """
    Read data as provided and return it as list of dictionaries.

    The function takes care of cleaning data, aligning it, etc. Note that this
    version works for the files *with* test deletion -- check
    `prepare_data_traint()` as well.
    """

    # Read both sources
    with open(test_file, encoding="utf-8") as handler:
        test = list(csv.DictReader(handler, delimiter="\t"))
    with open(solution_file, encoding="utf-8") as handler:
        solutions = defaultdict(dict)
        for entry in csv.DictReader(handler, delimiter="\t"):
            cogid = entry.pop("COGID")
            lects = {lect: val for lect, val in entry.items() if val}
            for lect, form in lects.items():
                solutions[cogid][lect] = form

    # Fill test data with the solutions (we will be able to detect which one is
    # test from the id, later)
    data = []
    for entry in test:
        cogid = entry["COGID"]
        entry_copy = dict(entry)
        entry_copy.update(solutions[cogid])
        data.append(entry_copy)

    return data


def write_mt(mt_data: List[dict], filename: str):
    """
    Write multitiers data to disk.
    """

    fields = ["ID", "DOCULECT", "TOKENS", "ALIGNMENT", "COGID"]
    with open(filename, "w", encoding="utf-8") as handler:
        writer = csv.DictWriter(handler, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(mt_data)


def extend_data(runargs: dict):
    """
    Script entry point.
    """

    # Prepare all training data; this also drop erroneous files in the release
    train_pattern = Path(runargs["source_dir"]) / "*" / "training*"
    train_files = glob.glob(str(train_pattern))
    train_files = [filename for filename in train_files if "-out.tsv" not in filename]
    if not train_files:
        raise ValueError("No training data was found -- is the path right?")

    for filename in train_files:
        output_path = train2mt(filename)
        if output_path.is_file():
            print(f"File `{output_path}` already exists, skipping.")
        else:
            logging.info(f"Extending `{filename}`...")
            with open(filename, encoding="utf-8") as handler:
                data = list(csv.DictReader(handler, delimiter="\t"))
                mt_data = extend_with_alignment(data)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_mt(mt_data, str(output_path))

    # Prepare test data
    test_pattern = Path(runargs["source_dir"]) / "*" / "test*"
    test_files = glob.glob(str(test_pattern))
    solutions_files = [filename.replace("test", "solutions") for filename in test_files]
    for test_f, solutions_f in zip(test_files, solutions_files):
        output_path = train2mt(test_f)
        if output_path.is_file():
            print(f"File `{output_path}` already exists, skipping")
        else:
            logging.info(f"Extending `{test_f}` and `{solutions_f}`...")
            data = prepare_data_test(test_f, solutions_f)
            mt_data = extend_with_alignment(data)
            write_mt(mt_data, str(output_path))


def build_tier_data(
    source: str, left: int = 1, right: int = 1, mapping: Tuple = ("SC",)
) -> pd.DataFrame:
    """
    Builds tier data from a wordlist.
    """

    # Read the wordlist and obtain a list of doculects
    wordlist = multitiers.read_wordlist(source)

    # Build and fit and encoder
    mapper_file = BASE_PATH / "etc" / "maniphono.csv"
    encoder = multitiers.MTEncoder(mapper_file=str(mapper_file))
    encoder.fit(wordlist, left=left, right=right, mapping=list(mapping))

    # Encode the wordlist
    data = encoder.transform(wordlist)

    return data


def build_dfs(runargs: dict):
    """
    Read aligned data and output dataframes with tiers.
    """

    # Get filenames to be transformed; the third line is a safety measure in
    # case of users running it without cleaning the output of a previous run
    data_files = glob.glob(str(BASE_PATH / "mtdata" / "*" / "training*"))
    data_files += glob.glob(str(BASE_PATH / "mtdata" / "*" / "test*"))

    for filename in sorted(data_files):
        logging.info(f"Building mtiers for `{filename}`...")
        mtiers = build_tier_data(
            filename,
            left=runargs["left"],
            right=runargs["right"],
            mapping=runargs["mapping"].split(","),
        )

        output_path = mt2df(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_file():
            print(f"File `{output_path}` already exists, skipping")
        else:
            mtiers.to_csv(
                output_path,
                sep="\t",
                index=False,
                float_format="%.4f",
                encoding="utf-8",
            )


def main(runargs: dict):
    """
    Script entry point.
    """

    extend_data(runargs)
    build_dfs(runargs)


def parse_args() -> dict:
    """
    Parse command-line arguments and return them as a dictionary.
    """

    parser = argparse.ArgumentParser(
        description="Extend ST2022 challenge data with multitier information."
    )
    parser.add_argument(
        "source_dir",
        type=str,
        # default="/home/tiagot/repos/sigtyp2022/ST2022/data",
        help="Path to the directory with the source files (ending in `ST2022/data`).",
    )
    parser.add_argument(
        "-l",
        "--left",
        type=int,
        default=3,
        help="Size of the left context (default: 3).",
    )
    parser.add_argument(
        "-r",
        "--right",
        type=int,
        default=3,
        help="Size of the right context (default: 3).",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="SC,PROSODY",
        help="List of mappings to add, separated by commas (default: `SC,PROSODY`).",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level.",
    )

    # Get the namespace dictionary, also for web interface compatibility
    runargs = parser.parse_args().__dict__

    return runargs


if __name__ == "__main__":
    # Parse command line arguments and set the logger
    args = parse_args()

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(level=level_map[args["verbosity"]])

    main(args)
