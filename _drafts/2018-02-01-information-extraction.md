---
layout: post
title: Information Extraction using NLTK
tags: NLP
---

Typical pipeline

1. Tokenize sentences
2. Tokenize words
3. Part of speech tagging
4. Named entity recognition
5. Relation extraction

# More particular

## 1. Tokenize sentences

Most text processing tasks will begin with a large volume of text, possibly in many smaller volumes such as a collection of news articles or tweets. The first step in processing this data is to tokenize the sentences as a starting point in breaking the data down into more easily digested portions. Assuming our data is a collection of news articles, we would start by dividing each datapoint from a full story to an list of sentences. Thus, a selection from this list of sentences might be the following:

>> Attorney General Jeff Sessions has named interim United States attorneys in Manhattan, one of the country’s most prominent federal prosecutors’ offices, replacing lawyers who had served as top deputies to Obama appointees.

## 2. Tokenize words

After tokenizing the sentences, we further tokenize these sentences down to the individual words.

>> ['Attorney', 'General', 'Jeff', 'Sessions', 'has', 'named', 'interim', 'United', 'States', 'attorneys', 'in', 'Manhattan', ',', 'one', 'of', 'the', 'countrys', 'most', 'prominent', 'federal', 'prosecutors', 'offices', ',', 'replacing', 'lawyers', 'who', 'had', 'served', 'as', 'top', 'deputies', 'to', 'Obama', 'appointees', '.']

## 3. Part of speech tagging

Most analysis of the words within a sentence requires us to first analyze the role the given word plays within the sentence. The best way to accomplish this is tagging words by their part of speech.

>> [('Attorney', 'NNP'), ('General', 'NNP'), ('Jeff', 'NNP'), ('Sessions', 'NNP'), ('has', 'VBZ'), ('named', 'VBN'), ('interim', 'JJ'), ('United', 'NNP'), ('States', 'NNPS'), ('attorneys', 'NNS'), ('in', 'IN'), ('Manhattan', 'NNP'), (',', ','), ('one', 'CD'), ('of', 'IN'), ('the', 'DT'), ('countrys', 'NN'), ('most', 'RBS'), ('prominent', 'JJ'), ('federal', 'JJ'), ('prosecutors', 'NN'), ('offices', 'NNS'), (',', ','), ('replacing', 'VBG'), ('lawyers', 'NNS'), ('who', 'WP'), ('had', 'VBD'), ('served', 'VBN'), ('as', 'IN'), ('top', 'JJ'), ('deputies', 'NNS'), ('to', 'TO'), ('Obama', 'NNP'), ('appointees', 'NNS'), ('.', '.')]

## 4. Named entity recognition

Now that we have identified the words by their parts of speech, we can more capably perform _named entity recognition_. At it's most basic, this is the process of identifying proper nouns such as people or specific locations, but can be expanded out to any kind of noun. This process is imperfect, but this may be an ideal outcome depending upon the intent.

>> [('OCCUPATION', 'Attorney General'), ('PERSON', 'Jeff Sessions'), 'has', 'named', 'interim', ('PLACE', 'United States'), 'attorneys', 'in', ('PLACE', 'Manhattan'), ',', 'one', 'of', 'the', 'countrys', 'most', 'prominent', 'federal', 'prosecutors', 'offices', ',', 'replacing', 'lawyers', 'who', 'had', 'served', 'as', 'top', 'deputies', 'to', ('PERSON', 'Obama'), 'appointees', '.']

## 5. Relation extraction

Now that the entities within the text have been identified, we can extract semantic relationships between these entities and their surrounding text. This process is critical to extracting useful, human-interpretable information from the text.

>> [(('OCCUPATION', 'Attorney General'), 'is', ('PERSON', 'Jeff Sessions')), (('PLACE', 'Manhattan'), 'is', ('one of the countrys most prominent federal prosecutors offices'))]