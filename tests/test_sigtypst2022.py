from pathlib import Path
from sigtypst2022 import (
        CorPaRClassifier, Baseline, download, prepare,
        load_cognate_file, write_cognate_file, 
        split_training_test_data, split_data, simple_align,
        predict_words, compare_words)
import tempfile


DATASETS = {
  "allenbai": {
      "subgroup": "subgroup", 
      "name": "Bai",
      "path": "lexibank/allenbai",
      "version": "v4.0",
      "cognates": ""
      },
    "listsamplesize": {
      "subgroup": "family",
      "name": "Indo-European",
      "path": "sequencecomparison/listsamplesize",
      "version": "v1.0",
      "cognates": "cogid_cognateset_id"}}


def data_path(*comps):
    return Path(__file__).parent.joinpath(*comps)


def test_download():
    with tempfile.TemporaryDirectory() as f:
        download(
                DATASETS,
                Path(f))

    



def test_prepare():
    with tempfile.TemporaryDirectory() as f:
        prepare(
                DATASETS,
                Path(f),
                data_path("cldf"))


def test_load_cognate_file():
    languages, sounds, data = load_cognate_file(
            data_path("data", "allenbai", "cognates.tsv"))
    assert len(languages) == 9

def test_write_cognate_file():
    with tempfile.TemporaryDirectory() as f:
        out = Path(f).joinpath("dummmy.tsv")
        languages = ["a", "b", "c"]
        data = {
                "1": {"a": "a", "b": "b"},
                "2": {"a": "b", "b": "c"}
                }
        write_cognate_file(languages, data, out)


def test_simple_align():

    out = simple_align(
            [["b", "a", "k"], ["b", "a"]],
            ["a", "b"],
            ["a", "b", "c"],
            training=False
            )
    assert len(out) == 3

    out = simple_align(
            [["b", "k"], ["b", "a", "k"]],
            ["a", "b"],
            ["a", "b", "c"],
            training=True
            )
    assert len(out) == 2


def test_baseline():

    bl = Baseline(data_path("data", "allenbai", "training-0.20.tsv")) 
    bl.fit()
    out = bl.predict(["Eryuan", "Heqing", "Jianchuan"], 
            [
                "x e ⁵⁵".split(),
                "xʰ ẽ ⁵⁵".split(),
                "x ẽ ⁵⁵".split()
                ],
            "Lanping")
    assert out[0] == "Ø"


def test_split_training_test_data():
    languages, sounds, data = load_cognate_file(
            data_path("data", "allenbai", "cognates.tsv"))
    training, test, solutions = split_training_test_data(
            data, languages)


def test_split_data():

    split_data(DATASETS, data_path("data"))


def test_predict_words():

    predict_words(
            data_path("data", "allenbai", "training-0.20.tsv"),
            data_path("data", "allenbai", "test-0.20.tsv"),
            data_path("data", "allenbai", "results-0.20.tsv")
            )


def test_compare_words():
    rep = compare_words(
            data_path("data", "allenbai", "results-0.20.tsv"),
            data_path("data", "allenbai", "solutions-0.20.tsv"),
            )
    assert len(rep) == 10
