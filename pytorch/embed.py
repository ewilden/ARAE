from collections import Counter
from itertools import chain
from itertools import groupby
from operator import itemgetter
from string import punctuation
import nltk
import numpy as np
from numpy.linalg import  norm
from scipy import sparse as sp
try:
  from config import *
except ImportError:
  from nlp.config import *


PUNCTUATION = set(punctuation)
NLP = None


def split_on_punctuation(document):
  '''tokenizes string by splitting on spaces and punctuation
  Args:
    document: string
  Returns:
    generator of strings
  '''

  for token in document.split():
    if len(token) == 1:
      yield token
    else:
      chunk = token[0]
      for char0, char1 in zip(token[:-1], token[1:]):
        if (char0 in PUNCTUATION) == (char1 in PUNCTUATION):
          chunk += char1
        else:
          yield chunk
          chunk = char1
      if chunk:
        yield chunk


def tokenize(documents, fast=True):
  '''tokenizes documents
  Args:
    documents: iterable of documents to tokenize
    fast: use 'split_on_punctuation' function instead of spacy tokenizer
  Returns:
    list of tokenized documents
  '''

  if fast:
    return [list(split_on_punctuation(document)) for document in documents]
  global NLP
  if NLP is None:
    import spacy
    NLP = spacy.load('en', parser=False)
  return [[token.text for token in NLP(document)] for document in documents]


def get_ngrams(document, n):
  '''gets ngrams from tokenized document
  Args:
    documents: list of strings
    n: uses all k-grams for k=1,...,n
  Returns:
    generator of ngrams
  '''

  return chain(*(nltk.ngrams(document, k) for k in range(1, n+1)))


def count_features(features):
  '''computes feature counts
  Args:
    features: iterable of featurized documents
  Returns:
    dict mapping features to counts
  '''

  return Counter(feature for document in features for feature in document)


def sort_ngrams(ngrams):
  '''sorts n-grams, first by k (for k=1,...,n), then alphabetically (by 'w1 ... wk')
    Args:
      ngrams: iterable of tuples
    Returns:
      sorted list of n-grams
    '''

  return [gram for k, kgrams in groupby(sorted(ngrams, key=len), key=len) for gram, merged in sorted(((gram, ' '.join(gram)) for gram in kgrams), key=itemgetter(1))]


def ngram_vocab(documents, n=1, min_count=1):
  '''gets vocabulary from tokenized documents
  Args:
    documents: iterable of iterables of strings
    n: use k-grams for all k=1,...,n
    min_count: minimum number of times feature must be appear to be included in vocabulary
  Returns:
    dict mapping gram (tuple of strings) to index
  '''

  counts = count_features(get_ngrams(document, n) for document in documents)
  return {gram: i for i, gram in enumerate(sort_ngrams(gram for gram, count in counts.items() if count >= min_count))}


def docs2bongs(documents, vocabulary=None, weights=None, default=1.0, **kwargs):
  '''computes Bag-of-n-Grams vectors from tokenized text documents
  Args:
    documents: iterable of lists of strings
    vocabulary: dict mapping gram (tuple of strings) to index (nonnegative int); if None will compute it automatically
    weights: dict mapping grams to weights; if None will compute unweighted BonGs
    default: default weight to assign a feature if it's not in weights; ignored if weights is None
    **kwargs: additional arguments passed to ngram_vocab; ignored if not vocabulary is None
  Returns:
    sparse (CSR) matrix of BonGs of size (len(documents), len(vocabulary))
  '''

  if vocabulary is None:
    vocabulary = ngram_vocab(documents, **kwargs)
  vocabset = set(vocabulary)
  n = len(max(vocabset, key=len))
  V = len(vocabulary)

  ngrams = ((gram for gram in get_ngrams(document, n) if gram in vocabset) for document in documents)
  rows, cols, values = zip(*((row, col, count) for (row, col), count in Counter((i, vocabulary[gram]) for i, document in enumerate(ngrams) for gram in document).items()))
  bongs = sp.coo_matrix((values, (rows, cols)), shape=(len(documents), V), dtype=FLOAT).tocsr()

  if not weights is None:
    diag = np.empty(V)
    for gram, i in vocabulary.items():
      diag[i] = weights.get(gram, default)
    bongs = bongs.dot(sp.diags(diag, 0))
  return bongs


def vocab2vecs(vocabulary, random=None, vectorfile=None, corpus='CC', objective='GloVe', dimension=300, unit=True):
  '''assigns vectors to words in the vocabulary
  Args:
    vocabulary: iterable of tuples or strings
    random: type ('Gaussian' or 'Rademacher') of random vectors to use; if None uses pretrained vectors
    vectorfile: text file of word embeddings; ignored if not random is None
    corpus: corpus used to train embeddings; ignored if not random is None or not vectorfile is None
    objective: objective used to train embeddings; ignored if not random is None or not vectorfile is None
    dimension: embedding dimension; ignored if not random is None or not vectorfile is None
  Returns:
    dict mapping each word in the vocabulary to an embedding; words with no embedding are not included in this dict
  '''

  vocabset = {word for gram in vocabulary for word in gram if type(gram) == tuple}.union(word for word in vocabulary if type(word) == str)

  if random is None:

    if vectorfile is None:
      vectorfile = VECTORFILES[corpus][objective][dimension]
    w2v = {}

    with open(vectorfile, 'r') as f:
      for line in f:
        index = line.index(' ')
        word = line[:index]
        if word in vocabset:
          w2v[word] = np.array([FLOAT(entry) for entry in line[index+1:].split()[:dimension]])
          if unit:
            w2v[word] /= norm(w2v[word])

  elif random.lower() == 'gaussian':
    w2v = {word: vec / norm(vec) for word, vec in ((word, np.random.normal(size=dimension).astype(FLOAT)) for word in vocabset)}

  elif random.lower() == 'rademacher':
    w2v = {word: (2.0*np.random.randint(2, size=dimension).astype(FLOAT)-1.0)/np.sqrt(dimension) for word in vocabset}

  else:
    raise(NotImplementedError)

  return w2v


def docs2vecs(documents, w2v=None, weights=None, default=1.0, **kwargs):
  '''computes document embeddings from tokenized text documents
  Args:
    documents: iterable of lists of strings
    w2v: dict mapping words to vectors; if None will compute this using vocab2vecs
    weights: dict mapping words to weights; if None will compute unweighted embeddings
    default: default weight to assign a word if it's not in weights; ignored if weights is None
    **kwargs: additional arguments passed to vocab2vecs; ignored if not w2v is None
  Returns:
    numpy matrix of sentence embeddings of size (len(documents), d)
  '''

  if w2v is None:
    w2v = vocab2vecs({word for document in documents for word in document}, **kwargs)
  if not weights is None:
    w2v = {word: weights.get(word, default)*vec for word, vec in w2v.items()}

  dimension = w2v[list(w2v.keys())[0]].shape[0]
  z = np.zeros(dimension, dtype=FLOAT)
  return np.vstack(sum((w2v.get(word, z) for word in document), z) for document in documents)


def vocab2mat(vocabulary, **kwargs):
  '''constructs matrix of word vectors from word-to-index mapping
  Args:
    vocabulary: dict with words/tuples as keys and integer indices as values
    **kwargs: additional arguments passed to vocab2vecs
  Returns:
    numpy matrix of shape (V, d)
  '''

  vocabulary = dict((word[0], i) if type(word) == tuple else (word, i) for word, i in vocabulary.items())
  w2v = vocab2vecs(vocabulary, **kwargs)
  dimension = w2v[list(w2v.keys())[0]].shape[0]
  matrix = np.zeros((len(vocabulary), dimension), dtype=FLOAT)
  for word, i in vocabulary.items():
    vector = w2v.get(word)
    if not vector is None:
      matrix[i] = vector
  return matrix


if __name__ == '__main__':

  import __init__
  from load import *

  docs, _ = sst('train')
  documents = tokenize(docs)
  vocab = ngram_vocab(documents)
  bow = docs2bongs(documents, vocab)
  embed_matvec = bow.dot(vocab2mat(vocab))
  embed_online = docs2vecs(documents)
  write('\tError:\t' + str(norm(embed_matvec - embed_online)) + '\n')
