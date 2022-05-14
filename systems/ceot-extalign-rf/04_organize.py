#!/usr/bin/env python3

"""
Organize data as expected by the competition structure.


This is a q&d script to copy and organize the script as expected by the
sigtyp2022 challenge. It is provided so that the results written
by default in the `output/` directory are reorganized in the way
the general evaluation script finds them where they are expected.
"""

import os
import shutil
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent

def main(model):
    # just provide the list of datasets to be quick
    structure = [
    ["training", ["mannburmish","hantganbangime","felekesemitic",
               "hattorijaponic","listsamplesize",
               "abrahammonpa","allenbai",
               "backstromnorthernpakistan","castrosui","davletshinaztecan",
               ]],
    ["surprise", ["bremerberta","wangbai","beidazihui","hillburmish",
               "deepadungpalaung","luangthongkumkaren","bodtkhobwa",
               "birchallchapacuran","kesslersignificance",
               "bantubvd"]]
    ]

    results = ["result-0.10.tsv", "result-0.20.tsv", "result-0.30.tsv",
              "result-0.40.tsv","result-0.50.tsv"]

    for collection, datasets in structure:
        for dataset in datasets:
            print(f"copying {dataset}...")
            dirname = BASE_PATH / collection / dataset
            os.makedirs(str(dirname),exist_ok=True)

            for result in results:
                source = str(BASE_PATH / "output" / model / dataset / result)
                target = str(BASE_PATH / collection / dataset / result)

                shutil.copyfile(source, target)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        model = "best"
    else:
        model = sys.argv[1]

    main(model)
