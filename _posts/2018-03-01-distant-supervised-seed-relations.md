---
layout: post
title: Distant Supervision Seed Relation Extraction
tags: NLP 
---

Relation extraction can be difficult without considerable and varied labeled data. This is where semi-supervised and unsupervised methods become extraordinarily useful as they are intended to be capable of functioning without this initial labeled data. There are tradeoffs in these approaches as they are more prone to error but in cases where labeled data is unavailable and it is not practical to obtain it, they may be the only viable option. 

One such strategy called _distant supervised seed relation extraction_ performs relation extraction by examining text data looking for relations similar to labeled relations. It will look for more examples of these relations within the text it is examining and use the labels to extract the entities and relations. 

# Intuition

The concept behind distant supervised seed relation extraction is to use a set of pre-labeled relations, referred to as the _seeds_, to generate patterns by which more instances of that particular relation can be extracted. Consider an example seed of "Edgar Allen Poe, born in 1809". We want to maximize our ability to match this pattern in a corpus so this seed relation could be generalized to "PER, born in YEAR". After removing stop words and punctuation is transformed into the relatively straightforward relation of "PER born YEAR" and this pattern could be used to extract other instances from a corpus such as "James Joyce, born in 1882" or even "Walter Scott was born in 1771" if the pattern recognition is made to be sufficiently flexible. This process is an example of "distant supervision" because while training data is being provided, it is practically being used to identify _similar_ patterns rather than _precise_ examples.

# Feature Representations

Once the seeds have been preprocessed, they may still need to be transformed in order to use them as features. The feature representations could include regular expressions, bag of words, syntactic features, or embeddings.

Using regular expressions is a relatively straightforward process of transforming the patters to a regular expression. The above example of "PER, born in YEAR" would then be transformed into something like "[a-zA-Z]+, born in \d{4}". This is a very simple and limited example and would be more useful if we made further adaptations such as supporting multiple words in the pattern for PER but perhaps limiting to only a sequence of 1 to 3 capitalized words as well as expanding the pattern for the date to support all date formats. Once these regular expressions have been generated, they can be implemented by a simple search of the corpus for matches.

Bag of words can also be used as a feature representation. This could be using the full text of the relation or n-gram segments of the relation. These methods can provide a more flexible approach in the case of n-grams, but also creates an inefficiently large sparse vocabulary necessarily processed in order to perform the relation identification as is the case for any bag of words implementation. The sparse vocabularies can easily become computationally prohibitive in the context of dealing with any a corpus of moderate lengths.

The syntactic features primarily used are the path between entities and the part of speech tags. Like the bag of words feature representations, these values will require transformation into a numerical vector space prior to being passed to the model and poses the same potential complications of doing so.

Embeddings can also be used and do not suffer from the computational limitations previously mentioned. When properly trained, the embeddings can effectively support a nearest neigbors matching process for a given phrase where ", born in" and "was born in" and "was born" will all be located within a relatively narrow multidimensional space, making identification computationally easy.

# Unsupervised Seed Generation

Relation extraction by seed relations is a very effective method but one of the downsides is that that it requires a large amount of labeled relation seeds in order to become viable. In order to overcome this problem, it can be further expanded to a wholly unsupervised implementation by automatically extracting relations found within text and using these as seeds. Because this process is automated and thus highly prone to error, some metric of suspicion should be applied to newly discovered patterns in order to determine whether or not they represent legitimate semantic relations. The simplest method to resolving this potential issue is to only keep patterns found some number of times within the text. For instance, a pattern may be identified but not fully trusted as valid until it has been found 5 more times with varying entities.

The process functions in the following order:

1. Perform named entity recognition
2. Extract potential entity relations based upon dependency paths
3. Generalize potential relations to generic patterns
4. Validate relation patterns found
5. Apply to text corpus

## 1. Perform named entity recognition

Here we want to initially identify all possible target entities for relation extraction.

If the spacy library is not already installed, this can be resolved by 

```
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en --force
```

```python
import spacy

# localize text data
text = u"Joyce was born in 41 Brighton Square, Rathgar, Dublin, into a middle-class family."

# perform NER
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

# print entities & labels
for ent in doc.ents:
    print(ent.text, ent.label_)
```

```
(u'Joyce', u'PERSON')
(u'41', u'CARDINAL')
(u'Brighton Square', u'FAC')
(u'Rathgar', u'GPE')
(u'Dublin', u'GPE')
```

## 2. Extract potential entity relations based upon dependency paths

Identifying the entities tells us what the sentence is about, but this list doesn't give us useful information by itself. Given this example text, an ideal relation outcome would be ('Joyce', born, '41 Brighton Square'), ('Joyce', born, 'Rathgar'), ('Joyce', born, 'Dublin') and ('Joyce', born, 'middle class family'). In the English language, these relationships can be identified by the verbs. The most basic example relations make this clear: "Joyce _was_ born in Dublin", "Tim Cook _is_ CEO", "Paris _is_ in France". At a high level, the task to accomplish is locating the subject verbs of the named entities and the subsequent nouns or prepositional phrases. 

```python
verbs = []

for possible_verb in doc:
    if possible_verb.pos_ == "VERB":
        verbs.append(possible_verb)

relations = []
noun_phrases = [nc for nc in doc.noun_chunks]

for verb in verbs:
    # look for grammatically related tokens
    lefts = [t for t in verb.lefts]
    rights = [t for t in verb.rights]
    if len(lefts) or len(rights):
        # build full relation phrase(s) into discrete structure(s)
        for right in rights:
            new_relation = []
            [new_relation.append(left) for left in lefts]
            new_relation.append(verb)
            # collect object phrase
            right_objects = [t for t in right.rights]
            for right_object in right_objects:
                for np in noun_phrases:
                    if np.root == right_object:
                        new_relation.append(right)
                        new_relation.append(np)
                        relations.append(new_relation)
                        break
print(relations)
```

```
[u'Joyce', u'was', u'born', u'in', u'41 Brighton Square]
[u'Joyce', u'was', u'born', u'into', u'a middle class family']
```

## 3. Generalize potential relations to generic patterns

Now that we have extracted potential relations, we want to reduce them to patterns and remove duplicates. In order to accomplish this, we need to remove specific instances of an entity type and replace it with a marker to identify the type of entity that was found. For instance, we need to transform the potential relation ((u'Joyce', u'PERSON'), 'was born in', (u'Dublin', u'GPE')) to (u'PERSON', u'was born in', u'GPE') so that we can identify all other instances of this pattern.

```python
relation_patterns = []

for relation in relations:
    new_pattern = []
    for token in relation:
        # check for entity, otherwise add to relation pattern
        if type(token.text) in entities.keys():
            new_pattern.append(entities[token.text])
        else:
            new_pattern.append(token)
    relation_patterns.append(new_pattern)
```

## 4. Validate relation patterns found

The potential relations now need to be validated which we will do by usage counts. For instance, if we have a relation that is only used once in a large corpus, the assumption is made that it is an invalid relation whereas the more frequently found potential relation is more likely to be valid. Here we are working with an unrealistically small corpus for demonstration purposes so we will skip this step.

## 5. Apply to text corpus

Here we would take our extracted viable patterns and search the corpus for more examples of these patterns. This set of relation patterns can effectively extract all relationships previously seen from future texts.