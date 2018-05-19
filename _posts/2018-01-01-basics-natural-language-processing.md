---
layout: post
title: Basics of Natural Lanugage Processing
---

Natural Language Processing, or NLP, is a branch of research within data science intending to programmatically interpret, understand, or generate data in the form of natural human language. This field most commonly attempts to further the state of the art of such practices as machine translation of text between languages, information extraction, summarization, sentiment analysis, speech recognition, and subject categorization. These technologies are most frequently interacted with by consumers in the form of virtual assistants most obviously, but also text search engines and autocomplete functionalities now seen on mobile phones.

# Basics

Text data is very often noisy, inherently unstructured, and riddled with the eccentricities present in all human language. Computers are incapable of dealing with text directly and we need to process text data in order to transform it into a format that the computer can understand and effectively process.

## Preprocessing

Preprocessing in the context of NLP involves a number of techniques commonly applied to the text data, called the _corpus_, prior to modeling. Due to the subtlety and nuance present in all human languages, it is necessary to reduce the complexity within the data in order to make this data more manageable by the computer.

First we need some text to work with. Fortunately NLTK provides datasets to work with. We will use text from Twitter for our examples.

```python
import nltk
nltk.download('twitter_samples')

tweet_ids = nltk.corpus.twitter_samples.fileids()[2]
tweet_texts = nltk.corpus.twitter_samples.strings(tweet_ids)
```

### Tokenization

The first step often taken when approaching a new text data set is _tokenization_. This is simply the process of breaking up a block of text into isolated component objects, or _tokens_.

```python
print(tweet_texts[0])

nltk.download('punkt')

tokenized_tweet = nltk.tokenize.word_tokenize(tweet_texts[0])
print(tokenized_tweet)
```

The most common token levels used is words and phrases. Phrase tokenization very often is done in conjunction with a dictionary of common phrases which ought to be interpreted together rather than as individual words. "Natural Language Processing", for instance, is a more informative token than splitting this into 3 individual tokens of "Natural", "Language", and "Processing". The process of tokenization supports further preprocessing methods as it will enable easier execution of logic upon each token within the text.

### Lexicon Standardization

Different tenses of a given word are often irrelevant to language modeling. For instance, a sentiment analysis application is unconcerned with whether the subject of a sentence is currently doing something, did it in the past, or intends to do it in the future. It is very likely that the expansion of a _lexicon_, or the vocabulary within the text data, which will greatly increase processing time and memory consumed will not improve the accuracy of the categorization. _Lexicon standardization_ is the process of replacing redundant representations of words or phrases with a root word in order to improve performance and simplify the data set.

This is often done by _stemming_ or _lemmatization_. _Stemming_ is a simple rule-based approach to stripping words of suffixes. For instance, stemming would replace all instances of "giving", "given" and "give" with "giv". _Lemmatization_ is a more complex and organized method to finding root words for a given token. While more complex, it is more thorough in that it would be able to perform a similar operation as stemming on words with differing root letters. For instance, lemmatization would be able to replace all instances of "have", "having", and "had" with "have" whereas stemming would not be able to equate "had" with "have". Both of these methods are practically intended to condense the appearances of redundant representations to a single instance, such as reducing all instances of "listen", "listens", "listened", and "listening" to "listen".

```python
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

print(ps.stem('crying'))
print(ps.stem('running'))
print(ps.stem('assumption'))

stemmed_tweet = [ps.stem(t) for t in tokenized_tweet]
print(stemmed_tweet)
```

```python
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

print(lem.lemmatize('wolves'))
print(lem.lemmatize('cars'))
print(lem.lemmatize('geese'))

lemmed_tweet = [lem.lemmatize(t) for t in tokenized_tweet]
```

### Object Standardization

The accuracy of synonymous can be highly subjective. Words that are popularly considered to be synonamous may present critical differences in another context. For instance, an NLP application designed to extract requirements from job advertisements would likely consider the words "ninja", "master", and "wizard" to all communicate the same idea of indicating that the applicant should have a high degree of skill, knowledge, and experience in the given technology and would thus consider these words to be synonamous. An NLP application designed to produce movie summaries would, on the other hand, be very concerned with the distinction between these words. This is but one example of a case of colloquial expressions complicating a dataset, which could also reduce the accuracy and performance of an NLP application by confusing it with multiple occurrences of seemingly differing representations all being used in the same manner. This same challenge is faced when dealing with of abbreviations (ie, "wtf", "lol", "dm"), hashtags, slang, etc.

The process of addressing these issues is known as _object standardization_ and is typically done by simple search and replace from a dictionary of tokens.

## Feature Modeling

Once the data set has been cleaned and preprocessed, it needs to be converted into a format that the computer can understand. Namely, the natural language needs to be translated into numbers which can be used as features for a learning algorithm. There are a variety of methods used to accomplish this.

### Syntactic Parsing

While the lexical content of a corpus is very informative, this very aspect of the text is only important to us for the purposes of interpretation as a consequence of the relationships within the content. For example, consider the former sentence. If one were to tokenize that sentence and randomize the order of the tokens, it would become extremely difficult to accurately interpret. Defining the parts of speech within the sentence, however, gives an easily interpretable picture of the sentence. This analysis of the grammar and the relationships between the words is known as _syntactic parsing_ and includes such methods as dependency trees and part of speech tagging.

_Dependency trees_ are a grammatical representation of a parsed sentence which recursively identifies the relationships between all lexical items within the sentence. Each relationship is is represented by way of a triplet of the dependent, the relation and the governor. A dependency tree provides a powerful and easily interpreted view into a given sentence since it provides the context of each lexical item. 

_Part of speech tagging_ is, as the name implies, the process of tagging each word within a sentence with the function and usage of it within the sentence. Part of speech tagging can be very helpful for the purposes of disambiguation, normalization, lemmatization, and stopword removal. Additionally, it can be used to generally improve word-based features by adding the context usage of the word to the feature.

```python
print(nltk.pos_tag(tokenized_tweet))
```

### Entities

The simplest and most obvious feature from the corpus are the _n-grams_ themselves. N-grams are word chunks where _n_ defines the number of words to be used. This feature extraction process results in a sparse matrix of one-hot encoded vectors where each column represents a token from the lexicon and each row represents an n-gram.

An n-gram where every word is an isolated input is a _1-gram_ or _unigram_ and is probably the most frequently used feature set extracted from data. This is known as _bag of words_ and results in every word being represented as a column of the sparse matrix. N-gram representations where $n > 1$ are often more informative then bag of words as these representations will examine phrases rather than invidual words thus retaining more context for individual observations than a unigram.

Sometimes it can be beneficial to direct an algorithm to very specific attributes of the corpus rather than letting it examine all of it indiscriminately. This can be accomplished with _named entity recognition_ which is the process of extracting named entities from the corpus such locations, people, products, etc. This is done by identifying and extracting all noun phrases from the corpus, then classifying these phrases into the target categories (locations, people or products), and finally disambiguation for validation of the phrase classification.

Additionally, entities can be used as features by modeling directly upon the _topics_ identified within the text. These topics can be surfaced by examining patterns of co-occurring words within the corpus.

```python
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')

print(nltk.ne_chunk(nltk.pos_tag(tokenized_tweet)))
```

### Statistical Features

There are a variety of statistical characteristics that can be used directly as text features. One such statistic which is often used for information retrieval problems is _term frequency - inverse document frequency_, or TF-IDF, as it represents a measure of token importance relative to the corpus. The _term frequency_ is a simple count of a given token within a document and the _inverse document frequency_ is the log of the ratio of total documents in the corpus to number of documents containing the token. The product of these two values are equal to the tf-idf. Count based metrics can also be used such as sentence, word, punctuation counts or counts of a specific set of words.

### Entity Embeddings

Currently the most popular method of extracting features from text data is word embeddings. Embeddings are dense vector representations of real numbers which can reduce, increase or guarantee constant dimensionality of an input space, which is a feature representation of latent factors with which certain properties can be represented by notions of distance. More practically, embeddings transforms words into a vector of values that are automatically trained. The trained embedding may provide a visualization which places “mood” near “happy” and “sad”, but far from “Brazil” and “walking” or even support mathematical operations on the text such as “king” - “man” + "woman" = “queen” and "Windows" - "Microsoft" + "Google" = "Android".

Consider this in contrast to a bag of words approach where each token is represented by a sparse vector with the length equal to the number of tokens in the lexicon. If the lexicon is to be words (as opposed to letters or phrases), the lexicon can rapidly increase in length to tens of thousands of words. In bag of words, th vector representation of each word would be a vector of this length, tens of thousands, and every value would be zero except one. This is obviously not an ideal use of memory, but also reduces the representation of a give token to this single dimension of present or not present. An embedding, on the other hand, might realistically represent a single word by a vector of length 50, and each element of this vector will contain a value. This expands the representation of the word to many dimensions, reduces the vector dimensionality of the word representation as well as surfaces latent characteristics of the word that could easily surface the rich subtleties in natural language which cannot be accurately modeled in a one-hot encoded vector.

These benefits have made word embeddings a very popular NLP tool. Two very popular embedding models are Word2Vec and GloVe.
