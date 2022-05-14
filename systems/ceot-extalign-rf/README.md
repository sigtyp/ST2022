# ST2022 submission "extendend alignment"

This directory contains the ST2022 using a multitiered strategy (List and Chacon 2015; Tresoldi et al. 2018)
for cognate prediction. It is the first public submission using such a strategy,
still tentative in many of its implementations, with the prediction provided by different
machine learning methods.

## Instructions

In any standard Python environment, the system's dependencies can be installed with:

```bash
$ pip install -r requirements.txt
```

It is highly recommended to install within a virtual environment.

### Build multitiered data

Multitiered data, both as human-intended CSVs and `pandas` dataframes, must first be generated,
populating the `mtdata` and `dfs` directories, respectively. This can be done with the command:

```bash
$ ./01_build_tier_data.py source_dir
```

`source_dir` is a mandatory path to the root data directory of the ST2022 challenge. Please
note that files in `mtdata` and `dfs` will *never* be overwritten by design, so that you
need to clean these directories in case they hold data already.

This step is not necessary for reproducing the submission, as the pre-compiled and reproducible
output files are distributed with this submission.

The `01_build_tier_data.py` script allows to specify a number of options in terms of the multitiers
to be added and the left and right contexts. This submission used the default parameter values.
More information can be obtained from the command-line with the `--help` flag.

### Train models

Classification models must be trained in order to perform prediction later. This is by far
the slowest and most computational-intensive step: with the default parameters,
processing will take days on a normal laptop. This would not be necessary in normal
usage, but was an important step for this submission also in consideration of
the limits in computational power.

The complete set of trained models, serialized with `joblib`, is available
[here](https://uppsala.box.com/s/r9e8ag12dvpc6q24ufv06fis5v1bappy) (6.4 GB). The reduced set of the best models, as found by cross-validation
during training, is available [here](https://uppsala.box.com/s/tukf6gbpzqyl89hz6dznvbfxfk5ywl33) (1.1 GB). *Please note that, due to the
nature of machine learning systems in Python, it cannot be guaranteed that
the serialized models will work as expected in other systems, even when
the same version of Python and the libraries is used.*
The models here distributed were computed with the libraries in the versions specified
in `requirements.txt`, Python version 3.6.8 (GCC 4.8.5 20150623 Red Hat 4.8.5-44)
on CentOS Linux (3.10.0-1160.59.1.el7.x86\_64).
Also note that, due to the stochastic nature of the training, it might be impossible to obtain
fully reproducible results across different machines or even across different runs in
the same machine. Results should nonetheless be always comparable.

For a quick experimentation with good results, you might want to train only random
forests (the `rf` model), with 2 layers of stratified sampling and about 50
iterations for the hyperparameter tuning. On most common laptops, such a process should
finish in a couple of hours.

When invoked, this step will populate the `classifiers`
directory with serialized classifiers (dumped to disk with the `joblib` library) and
training logs/performance evaluators. This can be done with the command:

```bash
$ ./02_train.py
```

The `02_train.py` script offers a number of configurations related to training, which can be
set from the command-line and whose list can be obtained with the `--help` flag. In this
submission, the system will be default use the same training process for all
datasets, tuning hyperparameters.

The most important parameters for the training are `--kfolds` (defaulting to 4), which
specifies the number of k-folds for the stratified cross-validation, and
`--trials` (defaulting to 10) with the number of trials for hyperparameter optimization.
As a rule of thumb, a dataset training will take `kfolds * trials` the amount of time
of a single classifier fitting.

This submission is distributed with models trained by refining hyperparameters with Optuna,
with data internally managed as Pandas dataframes.
This design was implemented so that it should be relatively easy to medium to experienced
programmers to implement other classification methods (such as different classification
methods as those offered by `sklearn`, or different methods methods for hyperparameter
optimization).

### Perform predictions

Once classifiers have been trained, predictions for the ST2022 challenge can be
obtained by running them on the test data. his can be done with the command:

```bash
$ ./03_run.py source_data_path
```

The script will generate the requested file, keeping the expected structure, in
the `output/` directory. Predictions are in the format requested by the challenge,
with dataset names as directories within model names, e.g.
`output/abrahammonpa/results-0.10.tsv`. The auxiliary `internal_report.py`
allows to identify the best models based on training data, which can be
copied as the `best` predicting models. 

A custom table comparing the results with the baseline,
if available, will also be generated for each model within its `output/`
subdirectory.

### Evaluation

Evalution of the performance can be performed using the script provided by the
organizers. The following commands will write to disk a summary table for both this
system and the baseline (remember to set `--datapath` and `--dataset` appropriately,
and any `--proportion` you might want). Precompiled reports are available in
the `reports` directory.

```bash
$ st2022 --evaluate --proportion=0.1 --all --datapath=data-surprise --datasets=datasets-surprise.json --test-path=output/best
```

The results of all evaluations are provided in the `reports/` directory.

### Prepare submission

To organize the results in the file structure expected by the challenge,
a final script can be used. It will split the results in `training/` and `surprise/`,
accoring to the collection each dataset belongs to. By default it will
take the `best` model, but other model names can be provided at the command line.

```bash
$ ./04_organize.py [model_name]
```

## Reproducibility

Random seeds can be set from the command-line. Note, however, there are a number of caveats to consider:
besides the fact that full reproducibility cannot be guaranteed unless the exact same version of Python and
all the underlying libraries are used in equivalent machines, when optimizing a hyperparameter study in distributed
or parallel mode there is inherent non-determinism. Second, some objective functions behave in non-deterministic
ways. In short, while seeding a seed *might* guarantee, in the same machine, results which are reproducible across
runs, you will probably be unable to exactly reproduce the results here reported, especially
with the more stochastic models.

To circumvent this problem, all intermediate results and, especially, the training parameters for the best models
are reported to disk. This guarantees that results, if not fully reproducible, will be different by neglectable
margins of difference. It also allows to reduce the computational time by at least two orders of
magnitude, as the intensive computational step of hyperparameter tuning is not necessary anymore.

## References

List, Johann-Mattis. 2014. *Sequence comparison in historical linguistics*. Düsseldorf University Press: Düsseldorf. 

List, Johann-Mattis; Chacon, Thiago. 2015. *Towards a Cross-Linguistic Database for Historical Phonology? A Proposal
for a Machine-Readable Modeling of Phonetic Context*. Historical Phonology and Phonological Theory Workshop,
48th annual meeting of the Societas Linguistica Europaea, Leiden. 

List, Johann-Mattis. 2019. *Automatic sound law induction (Open problems in computational diversity linguistics 3)*. The Genealogical World of Phylogenetic Networks. url: https://phylonetworks.blogspot.com/2019/04/automatic-sound-law-induction-open.html

List, Johann-Mattis; Forkel, Robert; Hill, Nathan W. 2022. *A New Framework for Fast Automated Phonological Reconstruction Using Trimmed Alignments and Sound Correspondence Patterns* (pre-print). arXiv. doi: 10.48550/ARXIV.2204.04619

Tresoldi, Tiago; Anderson, Cormac; List, Johann-Mattis. 2018. *Modelling sound change with the help of multi-tiered
sequence representations*. 48th Poznań Linguistic Meeting (PLM2018).


## Acknowledgements

The submission was prepared by Tiago Tresoldi (tiago.tresoldi@lingfil.uu.se). It was developed in the context of
the Cultural Evolution of Texts project at the Department of Linguistics and Philology of the Uppsala
University, with funding from the Riksbankens Jubileumsfond
(grant agreement ID: MXM19-1087:1).

The author thanks Michael Dunn and Yingqi Jing for facilitating his access to the computing facilities that
were necessary to prepare this submission.

The author has previously worked in a research group led by one of the organizers of this challenge, and the
submission develops an idea partly developed by the latter. There was, however, no contact between the author and
the organizers on the matter before the submission of data and code for this challenge.
