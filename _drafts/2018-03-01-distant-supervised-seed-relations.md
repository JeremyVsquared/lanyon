---
layout: post
title: Distant Supervision Seed Relation Extraction
tags: NLP 
---

Relation extraction can be difficult without considerable and varied labeled data. This is where semi-supervised and unsupervised methods become extraordinarily useful.

# Intuition

The concept behind distant supervised seed relation extraction is to use a set of pre-labeled relations, referred to as the _seeds_, to generate patterns by which more instances of that particular relation can be extracted. Consider an example seed of "Edgar Allen Poe, born in 1809". This seed relation could be generalized to "PER, born in YEAR". After removing stop words and punctuation is transformed into the relatively straightforward relation of "PER born YEAR" and this pattern could be used to extract other instances from a corpus such as "James Joyce, born in 1882". This process is called "distant supervision" because training data is still being provided but then used to identify similar patterns rather than precise examples.

# Feature Representations

Once the seeds have been preprocessed, they may still need to be transformed in order to use them as features. The feature representations could include regular expressions, bag of words, syntactic features, or embeddings.

Using regular expressions is a relatively straightforward process of transforming the patters to a regular expression. The above example of "PER, born in YEAR" would then be transformed into something like "[a-zA-Z ]+, born in \d{4}". Once these regular expressions have been generatd, they can be implemented by a simple search of the corpus for matches.

Bag of words - full text, n-grams

The syntactic features primarily used are the path between entities and the part of speech tags. Like the bag of words feature representations, these values will require transforms into a numerical vector space prior to being passed to the model.

# Unsupervised Seed Generation

This idea can be expanded further to a wholly unsupervised implementation by using relations found within text and using these as seeds. Because this process is automated and thus highly prone to error, some metric of suspicion should be applied to newly discovered patterns in order to determine whether or not they represent legitimate semantic relations. The simplest method to resolving this potential issue is to only keep patterns found some number of times within the text. For instance, a pattern may be identified but not fully trusted as valid until it has been found 5 more times with varying entities.

The process functions in the following order:

1. Perform named entity recognition
2. Extract potential entity relations based upon dependency paths
3. Iterate through potential relations, normalizing the text by removing stop words and punctuation
4. Count instances of similar relations, removing those that do not meet the threshold of instances
5. Generalize the identified relations and apply to text corpus

## 1. Perform named entity recognition

```python
import spacy

# localize text data
text = "..."

# perform NER
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

# print entities & labels
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## 3. Extract potential entity relations based upon dependency paths

```python

```

## 4. Iterate through potential relations, normalizing the text by removing stop words and punctuation

```python

```

## 5. Count instances of similar relations, removing those that do not meet the threshold of instances

```python

```

## 6. Generalize the identified relations and apply to text corpus

```python

```