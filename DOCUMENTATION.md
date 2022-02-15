# Documentation of the `st2022` Software Package

## 1 Installation

We recommend to install with `pip`, but in development mode.

```
$ cd ST2022
$ pip install -e .
```

## Initializing the Development Data

Development data is derived from CLDF datasts (https://cldf.clld.org) from the Lexibank repository (https://github.com/lexibank/lexibank-analysed). Participants of the shared task do not need to follow these steps, abut we document htem here for completeness.

The packages are stored in the file `datasets.json`. This file contains the development data. For the surprise data, we will add a similar file.

To download the 10 development datasets with GIT, you can use the commandline, after having installed the `st2022` package successfully:

```
$ st2022 --download --cldf-data=cldf-data --datapath=data --datasets=datasts.json
```

This will download the data and store the datasets in the folder `cldf-data` on your system. Since the datasets are GIT repositories themselves, we do not provide them along with this package, but since they are all versionized, you will have the same versions on your system as other users, if you download them with the command above.

To prepare the data by computing automated cognate judgments from those datasets which do not have cognate sets already, type:

```
$ st2022 --prepare --cldf-data=cldf-data --datapath=data --datasets=datasets.json --runs=10000
```

This will load the CLDF datasets, analyze them with LingPy (https://lingpy.org) if needed, to infer cognates automatically, and write them into the folder `data/DATASETID/cognates.tsv` in the tabular format which we use for our data representation. To make sure that we can compare the files with the original data, a file in LingPy's wordlist format is also added to the path `data/DATASETID/wordlist.tsv`, but it won't be needed to work on the shared task.

Finally, to split the datasets into a training and a test set, which allows you to develop your systems, just type:

```
$ st2022 --split --datapath=data --datasets=dataset.json --seed
```

This will prepeare test-training splits in five versions (proportions of 0.1, 0.2, 0.3, 0.4, and 0.5 retained for testing). It will produce three files per dataset and proportion, all stored in the folder `data/DATASETID`. A file `solutions-{PROP}.tsv` (e.g., `solutions-0.10.tsv`), containing the solutions, and no further entries in our tabular format with languages as columns and cognate sets as rows. A file `training-{PROP}.tsv` (`training-0.10.tsv`) containing the training data, and a file `test-{PROP}.tsv` (`test-0.20.tsv`), containing the training data. In the latter file, the words that should be predicted are indicated by a `?`.

Note that these two steps are already carried out as part of the release of the development data. So you can test them for curiosity, but there is no need to run them, since we provide all data in the folder `data`.

## 3 Analyzing the Data with the Baseline

The baseline is based on a study by List (2019) but was slightly adapted here, so that it works swiftly with the specific data and is also specifically targeted at word prediction. To run this baseline for all datasets with a proportion of 0.2, you just need to type:

```
$ st2022 --predict --proportion=0.2 --all --datapath=data
```

This will apply the baseline procedure to the data, and predict words for all test data files that indicate that they contain a proportion of 0.2 of the whole datasets (that is: 20%). The results are written to files `data/DATASETID/results-0.20.tsv`. 

# 4 Evaluating Results

We offer a rather straightforward routine to check your results against the solutions. 

```
$ st2022 --compare --prediction-file=data/allenbai/result-0.20.tsv --solution-file=data/allenbai/solutions-0.20.tsv
```

This will yield the following output:

``` 
Language       ED    ED (Normalized)    B-Cubed FS
----------  -----  -----------------  ------------
Eryuan      0.474              0.155         0.810
Heqing      1.005              0.321         0.704
Jianchuan   0.784              0.260         0.762
Lanping     1.242              0.392         0.656
Luobenzhuo  2.015              0.645         0.440
Qiliqiao    0.423              0.131         0.827
Xiangyun    0.820              0.270         0.740
Yunlong     0.536              0.171         0.797
Zhoucheng   0.330              0.108         0.859
TOTAL       0.848              0.273         0.733
```

The column ED yields the un-normalized edit distance between prediction and attested word or morpheme. The column ED (Normalized) is the normalized score, and the column B-Cubed FS provides B-Cubed scores, following the suggestion of List (2019b) for computing B-Cubed scores instead of edit distances, which rank between 1 (perfect agreement) and 0.

To compute the evaluation for the entire data, just type:

```
$ st2022 --evaluate --datapath=data --datasets=datasets.json --all --proportion=0.2
```

The result here will summarize the scores per dataset:

```
DATASET                       ED    ED (NORM)    B-CUBED FS
-------------------------  -----  -----------  ------------
felekesemitic              3.254        0.556         0.476
mannburmish                2.386        0.608         0.463
hattorijaponic             1.364        0.302         0.702
listsamplesize             3.922        0.694         0.345
abrahammonpa               0.702        0.131         0.809
allenbai                   0.848        0.273         0.733
backstromnorthernpakistan  1.034        0.222         0.791
castrosui                  0.316        0.078         0.907
constenlachibchan          6.336        0.951         0.363
davletshinaztecan          3.310        0.490         0.537
```

You can also evoke the evaluation by using the `sigtypst2022` package directly.

```python
>>> from sigtypst2022 import compare_words
>>> compare_words("data/allenbai/results-0.20.tsv", "data/allenbai/solution-0.20.tsv", report=False)
[['Eryuan', 0.4742268041237113, 0.15549828178694142, 0.8099492089351455],
 ['Heqing', 1.0051546391752577, 0.3213058419243986, 0.7036422938197315],
 ['Jianchuan', 0.7835051546391752, 0.2598797250859107, 0.7619652162932989],
 ['Lanping', 1.2422680412371134, 0.39175257731958746, 0.6564076959750814],
 ['Luobenzhuo', 2.015463917525773, 0.6451890034364266, 0.4395111764873446],
 ['Qiliqiao', 0.422680412371134, 0.13144329896907211, 0.8268112869423385],
 ['Xiangyun', 0.8195876288659794, 0.2701890034364262, 0.73975821377089],
 ['Yunlong', 0.5360824742268041, 0.17139175257731956, 0.7969673892455005],
 ['Zhoucheng', 0.32989690721649484, 0.10781786941580752, 0.8588180770427789],
 ['TOTAL', 0.847651775486827, 0.27271859488354333, 0.7326478398346787]]
```



# 5 Loading Data

Datasets are stored in the folder `data`. 
