"""
Write your wrapper here.
"""
from sigtypst2022 import sigtypst2022_path, load_cognate_file, write_cognate_file



if __name__ == "__main__":
    for prop in ["0.10", "0.20", "0.30", "0.40", "0.50"]:
        print("[i] analyzing proportion {0}".format(prop))
        training = sigtypst2022_path("data-surprise").glob("*/training-{0}.tsv".format(prop))
        predict = sigtypst2022_path("data-surprise").glob("*/test-{0}.tsv".format(prop))
        for f1, f2 in zip(training, predict):
            ds = str(f1).split("/")[-2]
            print("[i] analyzing dataset {0}".format(ds))

