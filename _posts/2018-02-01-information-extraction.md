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

## 2. Tokenize words

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

```python
def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token
 
        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                            if ner != 'O':
                                ner = ner.split('-')[0]
 
                            if tag in ('LQU', 'RQU'):   # Make it NLTK compatible
                                tag = "``"
 
                            standard_form_tokens.append((word, tag, ner))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]
            

import string
from nltk.stem.snowball import SnowballStemmer
 
 
def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


# create a useful tagger with the GMB corpus
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
 
 
class NamedEntityTagger(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


# instantiate the tagger
rdr = read_gmb(corpus_root)

data = list(rdr)
data_train = data[:int(len(data)* 0.9)]
data_test = data[int(len(data) * 0.9):]

# instantiate the named entity tagger
nec = NamedEntityTagger(data_train)

# here assuming a preexisting variable containing the POS tagged text
prsd_entities = nec.parse(pos_tagged_text)
```

>[('OCCUPATION', 'Attorney General'), ('PERSON', 'Jeff Sessions'), 'has', 'named', 'interim', ('PLACE', 'United States'), 'attorneys', 'in', ('PLACE', 'Manhattan'), ',', 'one', 'of', 'the', 'countrys', 'most', 'prominent', 'federal', 'prosecutors', 'offices', ',', 'replacing', 'lawyers', 'who', 'had', 'served', 'as', 'top', 'deputies', 'to', ('PERSON', 'Obama'), 'appointees', '.']

There are a variety of pretrained datasets which can be used to classify named entities such as the GMB, but a very simple, albeit less effective alternative is to identify entities by extracting capitalized noun phrases and concatenating continugous nouns. Without a dictionary or semantic reference, these entities would not be identified by semantic entity type but this could be determined with further processing of the context such as dependent phrases or surrounding text.

## 5. Relation extraction

Now that the entities within the text have been identified, it becomes possible to extract relationships between the entities and the surrounding words and phrases. This is typically done by way of a trained classifier or pattern identification, but can also be done in a semi-supervised or unsupervised manner by simply extracting the text by dependency path and subsequently validating the relations by support of multiple instances of related statements. The following is an ideal output of relation extraction upon the previously tagged sentence.

>[(('OCCUPATION', 'Attorney General'), 'is', ('PERSON', 'Jeff Sessions')), (('PLACE', 'Manhattan'), 'is', ('one of the countrys most prominent federal prosecutors offices'))]

It should be self-evident by examining the output that identifying these relationships is an essential process to extracting information from the text. These relationships actually _are_ the information we wish to extract from the text, but it is first necessary to identify the _named entities_ and relevant noun phrases in order to get them.
