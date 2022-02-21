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
$ st2022 --download --cldf-data=cldf-data --datapath=data --datasets=datasets.json
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

Baseline results have already been computed with this release and are available from the repository, so they do not need to be repeated, but they can be repeated for curiosity.

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
abrahammonpa               0.662        0.121         0.817
allenbai                   0.840        0.273         0.731
backstromnorthernpakistan  1.126        0.240         0.780
castrosui                  0.323        0.080         0.905
davletshinaztecan          3.310        0.490         0.537
felekesemitic              3.254        0.556         0.476
hantganbangime             2.572        0.600         0.436
hattorijaponic             1.364        0.302         0.702
listsamplesize             3.922        0.694         0.345
mannburmish                2.418        0.609         0.464
```

You can also evoke the evaluation by using the `sigtypst2022` package directly.

```python
>>> from sigtypst2022 import compare_words
>>> compare_words("data/allenbai/result-0.20.tsv", "data/allenbai/solutions-0.20.tsv", report=False)
[['Eryuan', 0.520618556701031, 0.16967353951890018, 0.8030411631412933],
 ['Heqing', 0.9587628865979382, 0.30541237113402053, 0.721044304182591],
 ['Jianchuan', 0.7628865979381443, 0.2512886597938145, 0.7685757306818528],
 ['Lanping', 1.2525773195876289, 0.4012027491408932, 0.6166076435275766],
 ['Luobenzhuo', 2.020618556701031, 0.6619415807560144, 0.44894428771106515],
 ['Qiliqiao', 0.39690721649484534, 0.12328178694158066, 0.8313532574468973],
 ['Xiangyun', 0.8350515463917526, 0.2753436426116839, 0.7372053334977589],
 ['Yunlong', 0.4948453608247423, 0.16022336769759443, 0.7985857919632154],
 ['Zhoucheng', 0.31958762886597936, 0.10524054982817864, 0.8562898366404019],
 ['TOTAL', 0.8402061855670103, 0.2726231386025201, 0.7312941498658502]]
```

# 5 Loading Data

As mentioned before, datasets are stored in the folder `data` in varying proportions. The format is very straightforward, as you can see from the following table for a part of the data in `data/listsamplesize/training-0.10.tsv`. 

COGID | dutch | english | french | german
--- | --- | --- | --- | ---
423 | s t eː n | s t əʊ n |  | ʃ t ai n
2049 | m aː n | m uː n |  | m oː n t
2062 | s t ɔ r m | s t ɔː m |  | ʃ t ʊ r m
2065 | r eː ɣ ə m + b oː x | r eɪ n + b əʊ |  | r eː ɡ ə n + b oː ɡ ə n
1368 | s x aː d yː w | ʃ æ d əʊ |  | ʃ a t ə n
1365 | d ɑu w | d j uː |  | t au
754 | r eː ɣ ə | r eɪ n |  | r eː ɡ ə n
768 | w eː r | w ɛ ð ə r |  | v ɛ t ə r
1211 | v yː r | f aɪ ə r |  | f ɔy ə r

You can see that the first column stores cognate sets, while the remaining columns provide languages. Word forms are provided in segmented form, by separating individual sounds by spaces. Transcriptions follow the B(road)IPA of the CLTS project (https://clts.clld.org). 

The followig table shows the structure of the test data (`data/listsamplesize/test-0.10.tsv`).

COGID | dutch | english | french | german
--- | --- | --- | --- | ---
1469-1 | ? | s æ n d | s ɑ b l | z a n t
2047-1 | ? | s ʌ n | s ɔ l ɛ j | z ɔ n ə
2054-1 | ? | θ ʌ n d ə r | t ɔ n ɛ ʀ | d ɔ n ə r
1375-1 | ? | l aɪ t | l y m j ɛ ʀ | l ɪ x t
760-1 | ? | w ɪ n d | v ɑ̃ | v ɪ n t
756-1 | ? | s n əʊ | n ɛ ʒ | ʃ n eː
726-1 | ? | m ʌ ð ə r | m ɛ ʀ | m ʊ t ə r
1351-1 | ? | ɡ r æ n d + f ɑː ð ə r | ɡ ʀ ɑ̃ + p ɛ ʀ | ɡ r oː s + f aː t ə r
1353-1 | ? | ɡ r æ n + m ʌ ð ə r | ɡ ʀ ɑ̃ + m ɛ ʀ | ɡ r oː s + m ʊ t ə r

Here, the question mark indicates that the word should be predicted. The cognate set identifier is now augmented by a numerical identifier for the language which should be predicted (1 indicates `dutch` in this case).

The file with the solutions has the same structure, but only lists the solutions.

COGID | dutch | english | french | german
--- | --- | --- | --- | ---
1469-1 | z ɑ n t |  |  | 
2047-1 | z ɔ n |  |  | 
2054-1 | d ɔ n d ə r |  |  | 
1375-1 | l ɪ x t |  |  | 
760-1 | w ɪ n t |  |  | 
756-1 | s n eː w |  |  | 
726-1 | m uː d ə r |  |  | 
1351-1 | x r oː t + f aː d ə r |  |  | 
1353-1 | x r oː t + m uː d ə r  | | |

To load these files, we recommend to use the `sigtypst2022` library.

```python
>>> from sigtypst2022 import load_cognate_file, write_cognate_file
>>> languages, sounds, data = load_cognate_file("data/allenbai/training-0.10.tsv")
>>> print(languages[0])
Eryuan
>>> print(len(sounds["t"]["Eryuan"]))
41
>>> print(sounds["t"]["Eryuan"][0])
['1083', 0]
>>> print(data[sounds["t"]["Eryuan"][0][0]])
{'Eryuan': ['t', 'ɔ', '⁴²'], 'Heqing': ['t', 'ɔu', '⁴²'], 'Jianchuan': ['t', 'õ', '⁴²'], 'Lanping': ['t', 'u', '⁴²'], 'Luobenzhuo': ['t', 'ao', '⁴²'], 'Qiliqiao': ['t', 'w', 'ɔ', '³²'], 'Xiangyun': ['t', 'w', 'ɔ', '⁴²'], 'Yunlong': ['t', 'o', '⁴²'], 'Zhoucheng': ['t', 'u', '⁴²']}
```

You can see from this example, that the `languages` are a list of all languages. Sounds are a dictionary with language names as keys, sounds as values, which link themselves to tuples consisting of cognate sets in which they occur along with the index in the word form.

The data displays the content of each row as a dictionary:

```python
>>> data["1083"]
{'Eryuan': ['t', 'ɔ', '⁴²'],
 'Heqing': ['t', 'ɔu', '⁴²'],
 'Jianchuan': ['t', 'õ', '⁴²'],
 'Lanping': ['t', 'u', '⁴²'],
 'Luobenzhuo': ['t', 'ao', '⁴²'],
 'Qiliqiao': ['t', 'w', 'ɔ', '³²'],
 'Xiangyun': ['t', 'w', 'ɔ', '⁴²'],
 'Yunlong': ['t', 'o', '⁴²'],
 'Zhoucheng': ['t', 'u', '⁴²']}
 ```

In order to write a dataset in this structure to file, you just pass the language list with the data and a path indicating where the file should be written to the function `write_cognate_file`:

```python
>>> write_cognate_file(languages, {"1083": data["1083"]}, "test.tsv")
```

The file will now only contain one row with data:

```
$ cat test.tsv
COGID	Eryuan	Heqing	Jianchuan	Lanping	Luobenzhuo	Qiliqiao	Xiangyun	Yunlong	Zhoucheng
1083	t ɔ ⁴²	t ɔu ⁴²	t õ ⁴²	t u ⁴²	t ao ⁴²	t w ɔ ³²	t w ɔ ⁴²	t o ⁴²	t u ⁴²
```



