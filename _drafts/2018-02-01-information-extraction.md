---
layout: post
title: Information Extraction using NLTK
tags: NLP
---

Some of the most exciting opportunities within _Natural Language Processing_ require information extraction. Summarizing the key ideas in a research paper, identifying procedures and prescriptions relevant to current symptoms in a medical record, question answering systems and true conversational user interfaces all fundamentally require the ability of the system to extract pertinent information from text data.

The typical information extraction pipeline goes something like the following:

1. Tokenize sentences
2. Tokenize words
3. Part of speech tagging
4. Named entity recognition
5. Relation extraction

This can change given different domains or goals, but these changes would be subtle such as adding stemming/porting, standardizing named entities, etc.

# More particular

## 1. Tokenize sentences

Most text processing tasks will begin with a large volume of text, possibly in many smaller volumes such as a collection of news articles or tweets. The first step in processing this data is to tokenize the sentences as a starting point in breaking the data down into more easily digested portions. Assuming our data is a collection of news articles, we would start by dividing each datapoint from a full story to an list of sentences. Thus, a selection from this list of sentences might be the following:

``` python
text = "Attorney General Jeff Sessions has named interim United States attorneys in Manhattan, one of the country’s most prominent federal prosecutors’ offices, replacing lawyers who had served as top deputies to Obama appointees."
```

2. Tokenize words

After tokenizing the sentences, we further tokenize these sentences down to the individual words.

```python
from nltk import word_tokenize
tokens = word_tokenize(text)
print(tokens)
```

> ['Attorney', 'General', 'Jeff', 'Sessions', 'has', 'named', 'interim', 'United', 'States', 'attorneys', 'in', 'Manhattan', ',', 'one', 'of', 'the', 'countrys', 'most', 'prominent', 'federal', 'prosecutors', 'offices', ',', 'replacing', 'lawyers', 'who', 'had', 'served', 'as', 'top', 'deputies', 'to', 'Obama', 'appointees', '.']

## 3. Part of speech tagging

Most analysis of the words within a sentence requires us to first analyze the role the given word plays within the sentence. The best way to accomplish this is tagging words by their part of speech.

```python
from nltk import pos_tag
pos_tagged = pos_tag(tokens)
print(pos_tagged)
```

>[('Attorney', 'NNP'), ('General', 'NNP'), ('Jeff', 'NNP'), ('Sessions', 'NNP'), ('has', 'VBZ'), ('named', 'VBN'), ('interim', 'JJ'), ('United', 'NNP'), ('States', 'NNPS'), ('attorneys', 'NNS'), ('in', 'IN'), ('Manhattan', 'NNP'), (',', ','), ('one', 'CD'), ('of', 'IN'), ('the', 'DT'), ('countrys', 'NN'), ('most', 'RBS'), ('prominent', 'JJ'), ('federal', 'JJ'), ('prosecutors', 'NN'), ('offices', 'NNS'), (',', ','), ('replacing', 'VBG'), ('lawyers', 'NNS'), ('who', 'WP'), ('had', 'VBD'), ('served', 'VBN'), ('as', 'IN'), ('top', 'JJ'), ('deputies', 'NNS'), ('to', 'TO'), ('Obama', 'NNP'), ('appointees', 'NNS'), ('.', '.')]

## 4. Named entity recognition

Now that we have identified the words by their parts of speech, we can perform _named entity recognition_.  Important information found within text is dependent upon the context of a given subject of a phrase. In order to extract this information, we must first determine what a fact pertains to, or it's subject. _Named entity recognition_ is how do this and is practically the process of identifying proper nouns such as people or specific locations, but can be expanded out to any kind of noun.

There are a variety of libraries and methods of performing this operation, and the performance of these vary depending upon circumstances, corpus, and subject domain. This process is imperfect, but the following may be an ideal outcome depending upon the intent.

>[('OCCUPATION', 'Attorney General'), ('PERSON', 'Jeff Sessions'), 'has', 'named', 'interim', ('PLACE', 'United States'), 'attorneys', 'in', ('PLACE', 'Manhattan'), ',', 'one', 'of', 'the', 'countrys', 'most', 'prominent', 'federal', 'prosecutors', 'offices', ',', 'replacing', 'lawyers', 'who', 'had', 'served', 'as', 'top', 'deputies', 'to', ('PERSON', 'Obama'), 'appointees', '.']

## 5. Relation extraction

Now that the entities within the text have been identified, it becomes possible to extract relationships between the entities and the surrounding words and phrases. The following is an ideal output of relation extraction upon the previously tagged sentence.

>[(('OCCUPATION', 'Attorney General'), 'is', ('PERSON', 'Jeff Sessions')), (('PLACE', 'Manhattan'), 'is', ('one of the countrys most prominent federal prosecutors offices'))]

It should be self-evident by examining the output that identifying these relationships is an essential process to extracting information from the text. These relationships actually _are_ the information we wish to extract from the text, but it is first necessary to identify the _named entities_ and relevant noun phrases in order to get them.