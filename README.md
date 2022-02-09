# SIGTYP 2022 Shared Task: Prediction of Cognate Reflexes

## 1 The Reflex Prediction Task

In historical-comparative linguistics, scholars typically assemble words from
related languages into *cognate sets*. In contrast to the notion of *cognates*
in didactics and synchronic NLP applications, cognate words -- the members of a
cognate set -- are commonly assumed to share a common origin **regardless** of
their meaning. In addition, cognate sets should not contain borrowed words.
Cognate words typically show so-called *regular sound correspondences*. This
means that one can define a mapping across the individual phoneme systems of
the individual languages. Thus, English *t* typically corresponds to a German
*ts* (compare *ten* vs. *zehn*), and English *d* corresponds to German *t*
(compare *dove* vs. *Taube*). The mappings often depend on certain contextual
conditions and may differ, depending on the position in which they occur in a
words. Due to regular sound correspondences, linguists can often predict fairly
well how the cognate counterpart of a word in one language might sound in
another language. However, prediction by linguists rarely takes only one
language pair into account.  The more *reflexes* (counterparts) a cognate set
has in different languages, the easier it is to predict reflexes in individual
languages.

In its simplest form, the data we need for the task of reflex prediction is a
table in which each column represents a different language and each row a
different cognate set. Whenever a reflex in a specific language is missing,
this reflex can in theory be predicted with the help of the remaining reflexes.
As an example, consider the following table showing reflexes of cognate sets in German, English, and Dutch.


Cognate Set | German | English | Dutch
--- | --- | --- | --- 
ASH | a ʃ ə | æ ʃ | ɑ s
BITE | b ai s ə n | b ai t | b ɛi t ə
BELLY | b au x | - | b œi k

Since the reflex for the BELLY cognate sets is missing in English, we could try and predict it from known correspondences to German and Dutch. In fact, some English dialects even seem to keep the form *bouk*, which would be the correct prediction for this missing reflex. 

When provided with more data of this kind, one can provide a model that would be able to predict an English form given a German and a Dutch form, as well as a German form, given a Dutch and an Englisch form, and so on. Note that not all cognate sets in real-life data will have reflexes for all words. Thus, we know about English *bouk* from dialect records, but without dialects or written sources from Middle English, we could only rely on prediction itself in order to guess how the word would sound if it would have been retained in the respective language.

Since predictions for words that have been completely lost cannot be evaluated directly, we will base our task on the prediction of *artificially excluded word forms*. Thus, we first take a dataset with cognates in a couple of related languages, and then artificially delete some of the words in the datasets, using varying proportions. When training a model to predict the missing word forms, we can just compare the predicted words directly with the words we have deleted automatically, as was done in one early study on word prediction ([List 2019](http://doi.org/10.1162/coli_a_00344)). 

The task of supervised phonological reconstruction can be seen as a special case of the reflex prediction task. While we predict reflexes in *any* language in the generic reflex prediction task, we predict one specific reflex of a cognate set, the form in the ancestral language in automated phonological reconstruction.

## 2 Previous Work on Reflex Prediction

There are quite a few studies on supervised phonological reconstruction now. Unfortunately, only a few of them provide their source code and original data. The following list is not meant to be exhaustive, and we appreciate it if you point us to studies missing here by opening an issue.

* Liviu Dinu and Alina Maria Ciobanu. 2014. Building a dataset of multilingual cognates for the Romanian lexicon. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 1038–1043, Reykjavik, Iceland.  European Language Resources Association (ELRA).
* Timotheus Adrianus Bodt and Johann-Mattis List. 2021.  Reflex prediction. a case study of Western Kho-Bwa.  Diachronica:1–38.
* Johann-Mattis List. 2019a. Automatic inference of sound correspondence patterns across multiple languages. Computational Linguistics, 45(1):137–161.
* Carlo Meloni, Shauli Ravfogel, and Yoav Goldberg.  2021. Ab antiquo: Neural proto-language recon- struction. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Tech- nologies, pages 4460–4473, Online. Association for Computational Linguistics.

## 3 Datasets and Data Formats

Our data is taken from the Lexibank repository which offers wordlists from 100 standardized datasets ([List et al. 2021](https://doi.org/10.21203/rs.3.rs-870835/v1). In the repository, a larger collection of datasets come with cognate sets provided by experts and with phonetic transcriptions which were standardized by the Lexibank team. This means that we will have quite a few datasets that we can offer for development and testing. The data is originally coded in CLDF formats ([Forkel et al. 2018](https://doi.org/10.1038/sdata.2018.205)). In order to ease the access to the data in simple tabular form (which we consider most suitable for the training of supervised reflex prediction workflows), we provide small wrappers that allow to access individual datasets from within Python scripts.

The following code has still not been fully implemented, but gives you an impression of what you will have to do once we open the shared task by providing access to the development data.

```python
from sigtypst2022 import load_data, get_training_sample
languages, data = load_data("allenbai")
training_sample, words_to_predict, solutions = get_training_sample(languages, data)
```
In this example, `languages` is a list of language names. The data is a dictionary of dictionaries with cognate set identifiers as key, and as value a dictionary with language names as key and segmented words as value.  
The `training_sample` is also a nested list, but with some entries excluded. The `words_to_predict` is a dictionary of dictionaries, in which cognate set identifiers are again the key and the value is a set of language names which shoud be predicted for the given cognate set. The solutions contain the words which have been excluded. 

When testing the methods on the unseen datasets, we provide the input data in the form of a CSV file in which those words which should be predicted are marked by an `?`, while missing entries (resulting from real gaps in the cognate sets) are simply left blank. 

We plan to provide different proportions of missing data, ranging from 5% up to 50%. This will allow users to check how robust their systems are with respect to differing degrees of missing data.  

# 4 Evaluation

The expected prediction result for a given reflex is a list of phonetic transcription symbols (we segment all words in our CLDF datasets into sound units). This prediction can be directly compared against the attested form, which was removed from the data when training the model. A common metric by which we can compare the predicted form with the attested form is the edit distance. Formally, the edit distance is identical with the Hamming distance between the alignment of two strings. Working with alignments should be preferred to working with an algorithm computing the edit distance alone, since alignments are very useful for error analysis. Furthermore, when working with algorithms that produce fuzzy predictions which might predict more than one probable sound in a given sound slot, or alternative word forms, it is easier to score these predictions based on alignments, as shown in the study of [Bodt and List (2021)](https://doi.org/10.1075/dia.20009.bod). 

An additional evaluation measure based on alignments was proposed by [List (2019b)](https://doi.org/10.1515/tl-2019-0016). This measure takes into account that an algorithm might in theory commit systematic errors, which might be overly penalized by the edit distance. This method, which computes the B-Cubed F-Scores between the aligned predictions and attested forms, has been ignored in most approaches to supervised reflex prediction, but we consider it nevertheless useful to include it into the evaluation scores to be reported, since it comes theoretically much closer to the idea of regular sound correspondences in classical historical linguistics. 

Our dedicated Python package for the shared task will allow to compute all evaluation measures mentioned above (edit distance in raw and normalized form, B-Cubed F-Scores) from TSV files which can be passed to the command line as input. For the purpose of development, scholars can also use these metrics directly from within their Python scripts.

As a baseline, we will provide the method by [List (2019)](http://doi.org/10.1162/coli_a_00344), for which we will make a new release of the [LingRex](https://github.com/lingpy/lingrex) Python package, where the method will be included.

# 5 Data for Development and Data for the Final Evaluation

Our development data, which users should use to test and design their models, consists of 10 CLDF datasets of varying size, language families, and time depths. Data for the development phase may contain datasets which have been used in previous studies. For the actual testing of different systems, we will use 10 CLDF datasets which have so far not been analyzed in previous studies. To make sure that there are no major flaws in these datasets, all of them will be thoroughly checked and analyzed with our baseline methods.

# 6 Participation

In order to participate in the shared task, we ask participants to write their code in a transparent way that can be directly tested. Testable code requires that all dependencies of the code package are listed and that the code has been tested in a virtual environment. We also ask for detailed instructions that help in installing the packages and explain how they should be used.

# 7 Concluding Remarks

In case of questions, we kindly ask to raise issues, which we will try to answer as quickly as possible. In order to guarantee that users use the same datasets and the same code basis to access the data, we will provide releases of this code base. If problems should be detected at a later stage, we can make a new release and ask all users to switch to the most recent version. 


