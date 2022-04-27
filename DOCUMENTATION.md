# Documentation of the `st2022` Software Package

## 1 Installation

We recommend to install with `pip`, but in development mode.

```
$ cd ST2022
$ pip install -e .
```

## 2 Initializing the Data

### 2.1 Development Data

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
$ st2022 --split --datapath=data --datasets=datasets.json --seed
```

This will prepeare test-training splits in five versions (proportions of 0.1, 0.2, 0.3, 0.4, and 0.5 retained for testing). It will produce three files per dataset and proportion, all stored in the folder `data/DATASETID`. A file `solutions-{PROP}.tsv` (e.g., `solutions-0.10.tsv`), containing the solutions, and no further entries in our tabular format with languages as columns and cognate sets as rows. A file `training-{PROP}.tsv` (`training-0.10.tsv`) containing the training data, and a file `test-{PROP}.tsv` (`test-0.20.tsv`), containing the training data. In the latter file, the words that should be predicted are indicated by a `?`.

Note that these two steps are already carried out as part of the release of the development data. So you can test them for curiosity, but there is no need to run them, since we provide all data in the folder `data`.

As a short cut, you can also use our Makefile and type:

```
$ make prepare-training
```

### 2.2 Surprise Data

The JSON file containing the links to the suprise datasets is called `data-surprise.json`, so in all commands, you have to replace `dataset.json` with `data-surprise.json`. While we download CLDF datasets into the same `cldf-data` folder (for convenience), we make a clear distinction, downloading data now to a `data-surprise` folder. 

```
$ st2022 --download --cldf-data=cldf-data --datapath=data-surprise --datasets=datasets-surprise.json
$ st2022 --prepare --cldf-data=cldf-data --datapath=data-surprise --datasets=datasets-surprise.json --runs=10000
$ st2022 --split --datapath=data-surprise --datasets=datasets-surprise.json --seed
```

As a short cut, you can also use our Makefile and type:

```
$ make prepare-surprise
```


## 3 Analyzing the Data with the Baseline

The baseline is based on a study by List (2019) but was slightly adapted here, as described in a forthcoming study by List et al. (forthcoming), so that it works swiftly with the specific data and is also specifically targeted at word prediction. To run this baseline for all datasets with a proportion of 0.2 on the development data, you just need to type:

```
$ st2022 --predict --proportion=0.2 --all --datapath=data
```

This will apply the baseline procedure to the data, and predict words for all test data files that indicate that they contain a proportion of 0.2 of the whole datasets (that is: 20%). The results are written to files `data/DATASETID/results-0.20.tsv`. 

Accordingly, for the suprise data, you type:

```
$ st2022 --predict --proportion=0.1 --all --datapath=data-surprise --datasets=datasets-surprise.json
```

Baseline results for development and surprise data have already been computed with this release and are available from the repository, so they do not need to be repeated, but they can be repeated for curiosity.

As a short cut, you can also use our Makefile and type:

```
$ make predict-training
$ make predict-surprise
```


# 4 Evaluating Results

We offer a rather straightforward routine to check your results against the solutions.

For the development data, for example, you can type: 

```
$ st2022 --compare --prediction-file=systems/baseline/training/allenbai/result-0.20.tsv --solution-file=data/allenbai/solutions-0.20.tsv
```

This will yield the following output:

``` 
Language       ED    ED (Normalized)    B-Cubed FS    BLEU
----------  -----  -----------------  ------------  ------
Eryuan      0.325              0.107         0.857   0.835
Heqing      0.691              0.221         0.751   0.685
Jianchuan   0.345              0.115         0.835   0.828
Lanping     1.067              0.338         0.628   0.523
Luobenzhuo  1.701              0.558         0.468   0.321
Qiliqiao    0.526              0.166         0.782   0.753
Xiangyun    0.670              0.219         0.715   0.685
Yunlong     0.541              0.180         0.773   0.749
Zhoucheng   0.325              0.106         0.856   0.842
TOTAL       0.688              0.223         0.741   0.691
```

For the surprise data, you can type accordingly:

```
$ st2022 --compare --prediction-file=systems/baseline/surprise/wangbai/result-0.10.tsv --solution-file=data-surprise/wangbai/solutions-0.10.tsv
```
This then yields the following output:

```
Language       ED    ED (Normalized)    B-Cubed FS    BLEU
----------  -----  -----------------  ------------  ------
Dashi       0.818              0.247         0.759   0.626
Ega         0.515              0.153         0.810   0.769
Enqi        0.530              0.158         0.818   0.764
Gongxing    0.773              0.232         0.810   0.665
Jinman      0.758              0.237         0.763   0.640
Jinxing     0.470              0.137         0.839   0.788
Mazhelong   0.545              0.162         0.815   0.762
ProtoBai    0.652              0.158         0.809   0.771
Tuoluo      0.576              0.159         0.810   0.763
Zhoucheng   0.561              0.183         0.825   0.734
TOTAL       0.620              0.183         0.806   0.728

```

The column ED yields the un-normalized edit distance between prediction and attested word or morpheme. The column ED (Normalized) is the normalized score, and the column B-Cubed FS provides B-Cubed scores, following the suggestion of List (2019b) for computing B-Cubed scores instead of edit distances, which rank between 1 (perfect agreement) and 0. The column BLEU provides BLEU scores (Papineni et al. 2002) in a new implementation that was tested to yield the scores as the NLTK implementation (`nltk.translation.sentence_bleu`).

To compute the evaluation for the entire data, just type:

```
$ st2022 --evaluate --datapath=data --datasets=datasets.json --all --proportion=0.1
```

The result here will summarize the scores per dataset:

```
DATASET                       ED    ED (NORM)    B-CUBED FS    BLEU
-------------------------  -----  -----------  ------------  ------
abrahammonpa               0.553        0.117         0.904   0.805
allenbai                   0.721        0.235         0.766   0.678
backstromnorthernpakistan  0.891        0.180         0.856   0.717
castrosui                  0.161        0.040         0.951   0.936
davletshinaztecan          2.074        0.331         0.644   0.520
felekesemitic              1.460        0.273         0.692   0.594
hantganbangime             1.312        0.326         0.624   0.537
hattorijaponic             0.914        0.194         0.799   0.725
listsamplesize             3.329        0.621         0.411   0.223
mannburmish                1.983        0.519         0.511   0.320
```

Accordingly, for the surprise data, you type:

```
$ st2022 --evaluate --proportion=0.1 --all --partition=surprise --datapath=data-surprise --datasets=datasets-surprise.json
```

And the resulting table looks like the following one.

```
DATASET                 ED    ED (NORM)    B-CUBED FS    BLEU
-------------------  -----  -----------  ------------  ------
bantubvd             1.120        0.255         0.788   0.619
beidazihui           1.102        0.301         0.730   0.582
birchallchapacuran   1.628        0.311         0.648   0.541
bodtkhobwa           0.492        0.203         0.757   0.721
bremerberta          1.725        0.317         0.705   0.510
deepadungpalaung     1.075        0.420         0.761   0.442
hillburmish          1.183        0.322         0.656   0.566
kesslersignificance  2.740        0.704         0.471   0.166
luangthongkumkaren   0.378        0.102         0.911   0.841
wangbai              0.620        0.183         0.805   0.729
```

You can also evoke the evaluation by using the `sigtypst2022` package directly.

```python
>>> from sigtypst2022 import compare_words
>>> compare_words("systems/baseline/training/allenbai/result-0.20.tsv", "data/allenbai/solutions-0.20.tsv", report=False)
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

As a short cut, you can also use our Makefile and type:

```
$ make evaluate-training
$ make evaluate-surprise
```

This will evaluate the data for all proportions.


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

# 6 Using Internal System Paths

The package also offers one path-function that you can use to get access to the data:

```python
from sigtypst2022 import sigtyst2022_path
training = sigtypst2022_path("data").glob("*/training-0.20.tsv")
```

This will yield a list `training` that contains all paths for the different test sets with 20% data held out.

# 7 Demo for the Integration of a System in the Shared Task

Our plan is to store systems and their results in specific folders, and to ask participants to add their data in pull-requests on GitHub, sharing results for training and for the surprise data. An example system has been added to the folder `systems`, called `corpar-svm`. This system contains a `run.py` file that can be invoked by typing `python run.py`, which will use the extended SVM classifier for the correspondence-based word prediction and phonological reconstruction framework by List et al. (forthcoming), which we use as a baseline in its original form, without a support vector machine and without specifically enriched alignments. Running the script with the argument `--surprise` will have it analyze the surprise data instead.

Results of this script (which itself is documented, so that users can check some of its major functions and see how to adapt their own code for their own systems), will be written to either a folder `surprise` or a folder `training`, both containing subfolders for all results on individual datasets. 

To check how well a system performed, the evaluation script of the `sigtypst2022` package contains a `--test-path` argument, which can be invoked as follows (assuming users have `cd`-ed into the main folder of the package):

```
$ st2022 --evaluate --datasets=datasets.json --datapath=data --test-path=systems/corpar-svm/training --all --proportion=0.10
```

The output is the same as we know from the test of the baseline results, but has now been applied to this extended baseline method using a support vector machine:

```
DATASET                       ED    ED (NORM)    B-CUBED FS    BLEU
-------------------------  -----  -----------  ------------  ------
abrahammonpa               0.344        0.067         0.914   0.888
allenbai                   0.643        0.209         0.777   0.713
backstromnorthernpakistan  0.589        0.116         0.888   0.802
castrosui                  0.131        0.031         0.960   0.949
davletshinaztecan          2.019        0.323         0.660   0.523
felekesemitic              1.386        0.261         0.719   0.611
hantganbangime             1.355        0.350         0.601   0.508
hattorijaponic             0.757        0.169         0.820   0.761
listsamplesize             2.679        0.489         0.528   0.372
mannburmish                1.702        0.457         0.559   0.385
```

Comparing this output with the results of the official baseline can be done as follows (also shown above):

```
st2022 --evaluate --datasets=datasets.json --datapath=data --all --proportion=0.10
```

```
DATASET                       ED    ED (NORM)    B-CUBED FS    BLEU
-------------------------  -----  -----------  ------------  ------
abrahammonpa               0.550        0.117         0.909   0.801
allenbai                   0.721        0.235         0.765   0.678
backstromnorthernpakistan  0.891        0.180         0.858   0.720
castrosui                  0.161        0.040         0.951   0.935
davletshinaztecan          2.074        0.331         0.644   0.520
felekesemitic              1.462        0.274         0.693   0.593
hantganbangime             1.311        0.326         0.621   0.540
hattorijaponic             0.936        0.198         0.789   0.719
listsamplesize             3.405        0.640         0.405   0.210
mannburmish                1.990        0.525         0.512   0.315
```

This shows that the extended SVM version of the baseline method enhances the results quite a lot, at least in this test of the training data.

In our demo, we provide all results when working on the SVM approach, both for the training data and for the suprise data. In addition, we offer a `template` folder in the `systems` folder that contains empty folders for the training and the surprise data as well as a rudimentary `run.py` script that users can use to add the code that one would need to apply their own systems to the data. 

Assuming that most participants code in Python, we kindly ask all participants of the shared task to try and prepare their systems in a similar form, as shown in our demo package. That means essentially:

1. there is a single folder that contains a script that runs the users' code and applies it too either the test data or the surprise data,
2. there is a file `requirements.txt` that contains all requirements that are needed to run this code in a virtual environment in Python (you can create the file by running `pip freeze > requirements.txt`),
3. there is a README.md file that explains how to run the code in question, and
4. the results are written into individual files, exactly as can be seen in our demo.

We understand that there are cases where users do not Python but other programming languages (R, Java, etc.) and that the `run.py` script is of little use in this case. In such a case, we still ask to provide a package of their system that contains specifically the structure of the output files, as we want to test those only with our software package, to make sure we have unified results. We then ask users to provide a clear description of their method, what programs need to be installed, how to run the code, etc., so that the shared task team can later try and apply their methods in order to make sure they run on another computer. 

To submit your systems to the shared task, we ask you to clone the ST2022 repository, add your system, and then make a pull request on GitHub, so that we can check the PR, maybe ask you for some modifications, and then add it to the repository. Note that the software for the system itself does not need to be included, it just needs to be accessible to us, and the system ideally contains the file that uses the systems to make the predictions.

