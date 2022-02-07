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

When provided with more data of this kind, one can provide a model that would be able to predict an English form given a German and a Dutch form, as well as a German form, given a Dutch and an Englisch form, and so on. Note that not all cognate sets in real-life data will have reflexes for all words. In the table, for example, we do not provide an English reflex for BELLY, since the word has to our knowledge not survived in the English language. 

The task of supervised phonological reconstruction can be seen as a special case of the reflex prediction task. While we predict reflexes in *any* language in the generic reflex prediction task, we predict one specific reflex of a cognate set, the form in the ancestral language in automated phonological reconstruction.

## Previous Work on Reflex Prediction

There are quite a few studies on supervised phonological reconstruction now. Unfortunately, only a few of them provide their source code and original data. 

+++ 

## Datasets and Data Formats

Our data is taken from the Lexibank repository which offers wordlists from 100 standardized datasets. In the repository, a larger collection of datasets come with cognate sets provided by experts and with phonetic transcriptions which were standardized by the Lexibank team. This means that we will have quite a few datasets that we can offer for development and testing. The data is originally coded in CLDF formats. In order to ease the access to the data in simple tabular form (which we consider most suitable for the training of supervised reflex prediction workflows), we provide small wrappers that allow to access individual datasets from within Python scripts.

```python
from sigtypst2022 import load_data

load_data("allenbai", feature_vectors=True)
```

* CLDF
* Python wrappers that present data in tabular form. 

