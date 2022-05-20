#!/usr/bin/env python3

"""
Run pretrained models on the data, evaluating results.
"""

# Import Python standard libraries
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import glob
import logging

# Import 3rd-party modules
import joblib
import pandas as pd
import tabulate

# Import other modules
import common
import my_sigtyp

BASE_PATH = Path(__file__).parent


def test2clf(filename: str, method: str) -> Path:
    """
    Build a clf filename from a corresponding test filename.
    """

    # todo: use pathlib
    tokens = filename.split("/")

    dataset = tokens[-2]
    train_file = tokens[-1].replace("test-", "training-")

    clf_file = BASE_PATH / "classifiers" / f"{dataset}.{train_file}.{method}.joblib"

    return Path(clf_file)


def test2df(filename: str) -> Path:
    """
    Build a df filename from a corresponding test filename.
    """

    # todo: use pathlib
    tokens = filename.split("/")

    dataset = tokens[-2]
    test_file = tokens[-1]

    df_file = BASE_PATH / "dfs" / dataset / test_file

    return Path(df_file)


# todo: use path to source
# todo: use pathlib
def test2ref(filename: str, source_path: str) -> Path:
    """
    Build a ref filename from a corresponding test filename.
    """

    tokens = filename.split("/")
    collection = tokens[-3]
    dataset = tokens[-2]
    filename = tokens[-1]

    # ref_file = f"/home/tiagot/repos/sigtyp2022/ST2022/{collection}/{dataset}/{filename}"
    ref_file = Path(source_path) / collection / dataset / filename

    return Path(ref_file)


# todo: use path to source
# todo: use pathlib
def test2baseline(filename: str, source_path: str) -> Path:
    """
    Build a baseline filename from a corresponding test filename.
    """

    tokens = filename.split("/")
    collection = tokens[-3]
    dataset = tokens[-2]
    filename = tokens[-1].replace("test-", "result-")

    # ref_file = f"/home/tiagot/repos/sigtyp2022/ST2022/{collection}/{dataset}/{filename}"
    ref_file = Path(source_path) / collection / dataset / filename

    return Path(ref_file)


# todo: use path to source
# todo: use pathlib
def test2solution(filename: str, source_path: str) -> Path:
    """
    Build a solution filename from a corresponding test filename.
    """

    tokens = filename.split("/")
    collection = tokens[-3]
    dataset = tokens[-2]
    filename = tokens[-1].replace("test-", "solutions-")

    # ref_file = f"/home/tiagot/repos/sigtyp2022/ST2022/{collection}/{dataset}/{filename}"
    ref_file = Path(source_path) / collection / dataset / filename

    return Path(ref_file)


def test2result(filename: str, method: str) -> Path:
    """
    Build a result filename from a corresponding test filename.
    """

    # todo: use pathlib
    tokens = filename.split("/")
    dataset = tokens[-2]
    filename = tokens[-1].replace("test-", "result-")

    return BASE_PATH / "output" / method / dataset / filename


def read_classifiers(filename: str):
    """
    Read classifiers serialized with joblib.
    """

    classifiers = joblib.load(filename)

    return classifiers


def sigtyp_output(prediction, ref_file, out_file, runargs: dict) -> bool:
    """
    Quick & dirty function for the output in the expected format.

    Returns a boolean indicating if anything was written to disk.
    """

    logging.info(f"Writing `{out_file}`...")

    # Build string representation for sigtyp evaluation
    words = {}
    for cogid, segments in prediction.items():
        word = " ".join([seg for seg in segments if seg != "-"])
        words[cogid] = word.strip("+").strip()

    # Read reference data
    with open(ref_file, encoding="utf-8") as handler:
        data = list(csv.DictReader(handler, delimiter="\t"))
        columns = list(data[0].keys())

    # Fill with the predicted words
    new_data = []
    for entry in data:
        new_entry = {}
        for key, value in entry.items():
            if value == "?":
                # todo: check why prediction is failing
                if entry["COGID"] in words:
                    new_entry[key] = words[entry["COGID"]]
                else:
                    new_entry[key] = "∅"
            elif key == "COGID":
                new_entry["COGID"] = value
            else:
                new_entry[key] = ""

        new_data.append(new_entry)

    # Output, creating the directory if necessary
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    if runargs["overwrite"] is False and Path(out_file).is_file():
        logging.warning(
            f"File `{out_file}` already exists (use `--overwrite` if necessary), skipping."
        )
        return False
    else:
        with open(out_file, "w", encoding="utf-8") as handler:
            writer = csv.DictWriter(handler, delimiter="\t", fieldnames=columns)
            writer.writeheader()
            writer.writerows(new_data)

    return True


def run_classifier_method(method: str, runargs: dict):

    # Get all testing files and the corresponding models, filtering out those that don't exist
    test_pattern_data = Path(runargs["source_dir"]) / "data" / "*" / "test*.tsv"
    test_pattern_data_surprise = (
        Path(runargs["source_dir"]) / "data-surprise" / "*" / "test*.tsv"
    )
    test_files = list(glob.glob(str(test_pattern_data)))
    test_files += list(glob.glob(str(test_pattern_data_surprise)))

    datasets = [
        {
            "test": Path(test_file),
            "clf": test2clf(test_file, method),
            "df": test2df(test_file),
            "ref": test2ref(test_file, runargs["source_dir"]),
            "result": test2result(test_file, method),
            "solution": test2solution(test_file, runargs["source_dir"]),
            "baseline": test2baseline(test_file, runargs["source_dir"]),
        }
        for test_file in sorted(test_files)
    ]
    datasets = [dataset for dataset in datasets if dataset["clf"].is_file()]

    # Run on all instances
    full_report = ""
    new_output = False
    # todo: dataset not a good name, rename it
    for dataset in datasets:
        logging.info(f"Predicting on {dataset['test']}...")

        # Obtain test mt and classifiers
        mt_test, _ = common.read_mtdf_data(str(dataset["df"]), drop_noninfo=False)
        classifiers = read_classifiers(str(dataset["clf"]))

        # Run over all doculects, also considering order
        prediction = defaultdict(list)
        for doculect_idx, doculect in enumerate(sorted(classifiers)):
            clf, features = classifiers[doculect]

            # Obtain X and y for running the classifiers
            if method not in ["lgb"]:
                X, y_real = common.get_X_y(
                    mt_test, doculect, method, features, encode=True
                )
                X = X.fillna(0)
            else:
                X, y_real = common.get_X_y(mt_test, doculect, method, features)

            # Collect the dataframe for generating full-word output; note that this
            # is specifically designed for the SIGTYP challenge, as we need
            # to set up a mask for cognates where the current doculect is missing
            mask = pd.notna(mt_test[f"segment_{doculect}"])
            for cogid, segment in zip(mt_test["cogid"][mask], clf.predict(X)):
                if int(cogid.split("-")[-1]) == doculect_idx + 1:
                    prediction[cogid].append(segment)

        # Output results from prediction, keeping track of whether at least one result was written to disk
        changed = sigtyp_output(prediction, dataset["ref"], dataset["result"], runargs)
        new_output = new_output or changed

        # If solutions and baseline are available and requesetd, compare results
        if not runargs["silent"]:
            if dataset["baseline"].is_file() and dataset["solution"].is_file():
                # Get our results and baseline's using sigtyp library
                mine = my_sigtyp.compare_words(
                    dataset["result"], dataset["solution"], report=False
                )
                baseline = my_sigtyp.compare_words(
                    dataset["baseline"], dataset["solution"], report=False
                )

                # Build and print results
                rows = [row_b + row_m[1:] for row_b, row_m in zip(baseline, mine)]
                report = tabulate.tabulate(
                    rows,
                    floatfmt=".4f",
                    headers=[
                        "Language",
                        "ED (base)",
                        "NED (base)",
                        "B2 (base)",
                        "BLEU (base)",
                        "ED (mine)",
                        "NED (mine)",
                        "B2 (mine)",
                        "BLEU (mine)",
                    ],
                )
                report = f"** {dataset['solution']}\n{report}\n\n\n"

                print(report)
                full_report += report

    # Write the full report if there were changes
    if new_output:
        with open(
            BASE_PATH / "output" / method / "full_report.txt",
            "w",
            encoding="utf-8",
        ) as handler:
            handler.write(full_report)


def main(runargs: dict):
    """
    Script entry point.
    """

    # TODO: Note that `best` is *not* a part of all
    if runargs["method"] == "all":
        for method in sorted(common.METHODS):
            run_classifier_method(method, runargs)
    else:
        run_classifier_method(runargs["method"], runargs)


def parse_args() -> dict:
    """
    Parse command-line arguments and return them as a dictionary.
    """

    parser = argparse.ArgumentParser(description="Run trained classifiers for ST2022.")
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="debug",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="all",
        choices=common.METHODS + ["all", "best"],
        help="Set the method for training.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite files if they exist (default is to keep them).",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Don't compare with baseline (no screen or file output).",
    )
    parser.add_argument(
        "source_dir",
        type=str,
        # default="/home/tiagot/repos/sigtyp2022/ST2022",  # todo: drop, note that it is note "data" only
        help="Path to the directory with the source files.",
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
