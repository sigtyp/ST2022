"""
Quick file for R&D for reporting scores for training.
"""

from pathlib import Path
import glob
from collections import defaultdict
import re
from pprint import pprint

BASE_PATH = Path(__file__).parent


def get_studies():
    # Load all trainings
    pattern = BASE_PATH / "classifiers" / "*.best_study.txt"
    filenames = sorted(glob.glob(str(pattern)))

    stats = defaultdict(dict)
    for filename in filenames:
        # Extract info with plain string manipulation
        tokens = filename.split("/")[-1].split(".")
        dataset = tokens[0]
        level = tokens[1] + "." + tokens[2]
        model = tokens[4]
        print(filename, [dataset, level, model])

        # Read the value for each -- dangerous eval
        with open(filename, encoding="utf-8") as handler:
            study = eval(handler.read())
            for doculect in study:
                stats[dataset, level, doculect][model] = eval(
                    study[doculect]["values"]
                )[0]

    # Show results
    for entry, values in sorted(stats.items()):
        print(entry)
        print(values)


def get_reports():
    # Load all trainings
    pattern = BASE_PATH / "output" / "*" / "full_report.txt"
    filenames = sorted(glob.glob(str(pattern)))

    stats = defaultdict(lambda: defaultdict(float))
    for filename in filenames:
        method = filename.split("/")[-2]

        with open(filename, encoding="utf-8") as handler:
            cur_file = None
            dataset = None

            for line in handler.readlines():
                if line.startswith("**"):
                    cur_file = line.split(" ")[1].strip()
                    dataset = "/".join(cur_file.split("/")[-2:])
                if line.startswith("TOTAL"):
                    tokens = re.sub(r"\s+", " ", line).split()
                    b2_base = float(tokens[3])
                    ned_base = float(tokens[2])
                    b2_mine = float(tokens[-2])
                    ned_mine = float(tokens[-3])
                    score = ((b2_mine - b2_base) + (ned_base - ned_mine)) / 2

                    stats[dataset][method] = score

    # pretty print
    pprint(stats)


def main():
    get_studies()
    get_reports()  # only for development


if __name__ == "__main__":
    main()
