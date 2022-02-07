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

In its simplest form, the data we need for the task of reflex prediction is a table in which each column represents a different language and each row a different cognate set. Whenever a reflex in a specific language is missing, this reflex can in theory be predicted with the help of the remaining reflexes.  

+++ table +++

The task of supervised phonological reconstruction can be seen as a special case of the reflex prediction task. While we predict reflexes in *any* language in the generic reflex prediction task, we predict one specific reflex of a cognate set, the form in the ancestral language in automated phonological reconstruction.

## Datasets and Data Formats

* CLDF
* Python wrappers that present data in tabular form. 

