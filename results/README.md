# Results for the Comparison

## General Summary on the Results

Detailed results are listed below for both the Training and the Surprise data. We summarize the major results of the shared task in this section.

The following table shows the winning system and the scores for the respective task and the respective proportion in the surprise data.

Proportion | Edit Distance | Edit Distance (Normalized) | B-Cubed Scores | BLEU Scores
--- | --- | --- | --- | ---
0.10 |  0.92 / Mockingbird-I1 | 0.24 / Mockingbird-I1 | 0.77 / Mockingbird-I1 | 0.66 / Mockingbird-I1 
0.20 | 1.04 / Mockingbird-I1 | 0.26 / Mockingbird-I1 | 0.71 / Mockingbird-I1 | 0.63 / Mockingbird-I1
0.30 | 1.18 / Mockingbird-I1 | 0.29 / Mockingbird-I1 | 0.67 / Mockingbird-I1 | 0.61 / Mockingbird-I1
0.40 | 1.27 / Mockingbird-I1 | 0.32 / Mockingbird-I1 | 0.64 / Mockingbird-I1 | 0.57 / Mockingbird-I1
0.50 | 1.47 / Mockingbird-I1 | 0.35 / Mockingbird-I1 | 0.62 / CrossLingferenceJulia | 0.53 / Mockingbird-I1

The overall winner of the task is the system I1 by the Mockingbird team. However, we can see that with sparser data for the training of the system available, the system Julia by the CrossLingference team comes closer in performance, outperforming the I1 system with respect to the B-Cubed evaluation measure.

## Detailed Results for all Datasets

Here, we list the results in the form of plots.
We list results for both Training and Surprise data. Training data is merely for checking if systems greatly differ for some reason, with respect to their results from the performance on the surprise data, which might warrant to double check with the code of the system. In some cases, the teams also could not provide all data for the individual training sets, due to time limitations and for other reasons. In these cases, the results in the tables have the value 0 in all scores.

To create the systems, you can most conveniently use our Makefile:

```
$ make compare-systems-training
$ make compare-systems-surprise
```

This yields as the output individual results in tabular form, which you can find in the files [results-training-0.10.md](results-training-0.10.md), 
[results-training-0.20.md](results-training-0.20.md),
[results-training-0.30.md](results-training-0.30.md),
[results-training-0.40.md](results-training-0.40.md), and
[results-training-0.50.md](results-training-0.50.md).

For the surprise data, the results can then be found in the files [results-surprise-0.10.md](results-surprise-0.10.md), 
[results-traning-0.20.md](results-surprise-0.20.md),
[results-traning-0.30.md](results-surprise-0.30.md),
[results-traning-0.40.md](results-surprise-0.40.md), and
[results-traning-0.50.md](results-surprise-0.50.md), respectively.

The resulting plots are also listed below.

### Results for the Training Partition (Proportion 0.10)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.3407       0.2835        0.7156  0.6057
Baseline-Baseline-SVM   1.1563       0.2435        0.7435  0.6577
CEoT-Extalign-RF        1.1469       0.2573        0.7313  0.6418
CrossLingference-Julia  1.2778       0.2785        0.7399  0.6197
Leipzig-Transformer     0.0000       0.0000        0.0000  0.0000
Mockingbird-I1          1.0528       0.2255        0.7447  0.6805
Mockingbird-N1-A        1.2533       0.2674        0.7152  0.6316
Mockingbird-N1-B        1.3053       0.2805        0.7042  0.6149
Mockingbird-N1-C        1.2906       0.2804        0.6929  0.6165
Mockingbird-N2          1.3686       0.2894        0.6823  0.6000
```

![0.10](training-0.10.png)

### Results for the Training Partition (Proportion 0.20)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.3857       0.2988        0.6735  0.5876
Baseline-Baseline-SVM   1.2190       0.2654        0.6982  0.6289
CEoT-Extalign-RF        1.2254       0.2822        0.6883  0.6068
CrossLingference-Julia  1.2842       0.2831        0.7066  0.6130
Leipzig-Transformer     0.0000       0.0000        0.0000  0.0000
Mockingbird-I1          1.0966       0.2382        0.7082  0.6661
Mockingbird-N1-A        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-B        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-C        0.0000       0.0000        0.0000  0.0000
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.20](training-0.20.png)

### Results for the Training Partition (Proportion 0.30)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.4626       0.3128        0.6525  0.5698
Baseline-Baseline-SVM   1.3959       0.3070        0.6534  0.5818
CEoT-Extalign-RF        1.3607       0.3118        0.6499  0.5710
CrossLingference-Julia  1.3194       0.2894        0.6879  0.6055
Leipzig-Transformer     0.0000       0.0000        0.0000  0.0000
Mockingbird-I1          1.1297       0.2501        0.6902  0.6498
Mockingbird-N1-A        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-B        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-C        0.0000       0.0000        0.0000  0.0000
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.30](training-0.30.png)

### Results for the Training Partition (Proportion 0.40)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.5184       0.3242        0.6354  0.5515
Baseline-Baseline-SVM   1.5902       0.3451        0.6173  0.5365
CEoT-Extalign-RF        1.5641       0.3572        0.6070  0.5156
CrossLingference-Julia  1.3342       0.2922        0.6760  0.6007
Leipzig-Transformer     0.0000       0.0000        0.0000  0.0000
Mockingbird-I1          1.2121       0.2693        0.6686  0.6240
Mockingbird-N1-A        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-B        0.0000       0.0000        0.0000  0.0000
Mockingbird-N1-C        0.0000       0.0000        0.0000  0.0000
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.40](training-0.40.png)

### Results for the Training Partition (Proportion 0.50)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.7541       0.3697        0.5978  0.5001
Baseline-Baseline-SVM   1.9828       0.4206        0.5530  0.4489
CEoT-Extalign-RF        1.8933       0.4230        0.5437  0.4398
CrossLingference-Julia  1.3762       0.2993        0.6647  0.5914
Leipzig-Transformer     0.0000       0.0000        0.0000  0.0000
Mockingbird-I1          1.4034       0.3088        0.6307  0.5787
Mockingbird-N1-A        1.6024       0.3391        0.5907  0.5362
Mockingbird-N1-B        1.6122       0.3445        0.5804  0.5309
Mockingbird-N1-C        1.6333       0.3530        0.5711  0.5215
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.50](training-0.50.png)

### Results for the Surprise Partition (Proportion 0.10)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.2095       0.3119        0.7231  0.5716
Baseline-Baseline-SVM   1.0189       0.2625        0.7626  0.6387
CEoT-Extalign-RF        1.0377       0.2763        0.7475  0.6243
CrossLingference-Julia  1.4804       0.3929        0.7251  0.4793
Leipzig-Transformer     1.3971       0.3716        0.6441  0.5083
Mockingbird-I1          0.9201       0.2431        0.7673  0.6633
Mockingbird-N1-A        1.0223       0.2568        0.7604  0.6479
Mockingbird-N1-B        1.0437       0.2625        0.7572  0.6398
Mockingbird-N1-C        1.1263       0.2867        0.7302  0.6115
Mockingbird-N2          1.2095       0.3135        0.7054  0.5744
```

![0.10](surprise-0.10.png)

### Results for the Surprise Partition (Proportion 0.20)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.3253       0.3361        0.6680  0.5412
Baseline-Baseline-SVM   1.1723       0.2928        0.7067  0.5985
CEoT-Extalign-RF        1.2208       0.3175        0.6798  0.5709
CrossLingference-Julia  1.4954       0.3912        0.6882  0.4760
Leipzig-Transformer     1.6518       0.4225        0.5529  0.4508
Mockingbird-I1          1.0413       0.2648        0.7120  0.6326
Mockingbird-N1-A        1.1512       0.2825        0.7011  0.6138
Mockingbird-N1-B        1.1726       0.2901        0.6910  0.6054
Mockingbird-N1-C        1.2196       0.3051        0.6669  0.5841
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.20](surprise-0.20.png)

### Results for the Surprise Partition (Proportion 0.30)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.4354       0.3556        0.6372  0.5195
Baseline-Baseline-SVM   1.3713       0.3310        0.6565  0.5554
CEoT-Extalign-RF        1.4038       0.3525        0.6331  0.5286
CrossLingference-Julia  1.6116       0.4130        0.6508  0.4503
Leipzig-Transformer     1.8206       0.4552        0.5080  0.4152
Mockingbird-I1          1.1762       0.2899        0.6717  0.6059
Mockingbird-N1-A        1.2565       0.3119        0.6557  0.5779
Mockingbird-N1-B        1.2712       0.3103        0.6531  0.5792
Mockingbird-N1-C        1.3009       0.3215        0.6343  0.5636
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```
![0.30](surprise-0.30.png)

### Results for the Surprise Partition (Proportion 0.40)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.6821       0.4011        0.6001  0.4717
Baseline-Baseline-SVM   1.6159       0.3891        0.5990  0.4903
CEoT-Extalign-RF        1.5695       0.3960        0.5805  0.4773
CrossLingference-Julia  1.6059       0.4112        0.6411  0.4473
Leipzig-Transformer     1.9890       0.4987        0.4640  0.3755
Mockingbird-I1          1.2725       0.3162        0.6428  0.5724
Mockingbird-N1-A        1.4542       0.3521        0.6294  0.5293
Mockingbird-N1-B        1.3618       0.3349        0.6212  0.5466
Mockingbird-N1-C        1.4353       0.3547        0.5999  0.5228
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.40](surprise-0.40.png)

### Results for the Surprise Partition (Proportion 0.50)

```
SYSTEM                      ED    ED (NORM)    B-Cubed FS    BLEU
----------------------  ------  -----------  ------------  ------
Baseline-Baseline       1.8889       0.4445        0.5617  0.4265
Baseline-Baseline-SVM   1.9330       0.4619        0.5371  0.4204
CEoT-Extalign-RF        1.8434       0.4576        0.5194  0.4128
CrossLingference-Julia  1.6794       0.4274        0.6193  0.4296
Leipzig-Transformer     2.1791       0.5422        0.4244  0.3337
Mockingbird-I1          1.4170       0.3518        0.6050  0.5337
Mockingbird-N1-A        1.5527       0.3800        0.5959  0.4934
Mockingbird-N1-B        1.5066       0.3734        0.5864  0.4989
Mockingbird-N1-C        1.5818       0.3950        0.5610  0.4749
Mockingbird-N2          0.0000       0.0000        0.0000  0.0000
```

![0.50](surprise-0.50.png)


