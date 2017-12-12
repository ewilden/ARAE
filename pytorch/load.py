import csv
import os
import sys
from unicodedata import normalize
try:
  from config import *
except ImportError:
  from nlp.config import *


FILEDIR = os.path.dirname(os.path.realpath(__file__)) + '/'
DATADIR = FILEDIR + 'data/'
PYTHONVERSION = sys.version[0]


def write(msg):
  '''writes to std out
  Args:
    msg: string
  Returns:
    length of msg
  '''

  sys.stdout.write(msg)
  sys.stdout.flush()
  return len(msg)


def csv2clf(filename):
  '''loads CSV file of form label\tdocument
  Args:
    filename: string of CSV filepath
  Returns:
    [list of labels, list of documents]
  '''

  with open(filename, 'r') as f:
    if PYTHONVERSION == '3':
      return list(zip(*(row[::-1] for row in csv.reader(f, delimiter='\t'))))
    return list(zip(*((unicode(normalize('NFKD', unicode(document, 'utf-8')).encode('ascii', 'ignore')), label) for label, document in csv.reader(f, delimiter='\t'))))


def sst(partitions=['train', 'test'], binary=True):
  '''loads Stanford Sentiment Treebank sentiment classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
    binary: load binary classification dataset instead of multi-label
  Returns:
    ((list of labels, list of documents) for each partition)
  '''

  if not binary:
    raise(NotImplementedError)
  if type(partitions) == str:
    return csv2clf(DATADIR+'sst_'+partitions+'.csv')
  return [csv2clf(DATADIR+'sst_'+partition+'.csv') for partition in partitions]


def imdb(partitions=['train', 'test']):
  '''loads Internet Movie Database sentiment classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of labels, list of documents) for each partition)
  '''

  if type(partitions) == str:
    return csv2clf(DATADIR+'imdb_'+partitions+'.csv')
  return [csv2clf(DATADIR+'imdb_'+partition+'.csv') for partition in partitions]


# NOTE: SkipThoughts only works in Python 2.
def skipthoughts(directional='uni'):
  '''initializes skip thoughts document encoder
  Args:
    directional: 'uni' or 'bi' - directional encoding
  Returns:
    EncoderManager object
  '''

  sys.path.append(FILEDIR)
  from skip_thoughts import configuration
  sys.path.append(FILEDIR+'skip_thoughts/')
  from skip_thoughts import encoder_manager
  encoder = encoder_manager.EncoderManager()
  encoder.load_model(configuration.model_config(bidirectional_encoder=directional=='bi'), **SKIPTHOUGHTS[directional])
  return encoder


if __name__ == '__main__':

  import __init__
  from learn import *

  (dtrain, ltrain), (dtest, ltest) = imdb()
  ltrain = np.array(ltrain)
  ltest = np.array(ltest)
  write('\tIMDB (binary)\t\tAcc.\n')

  root = '\t\rSkipThoughts (uni):\t'
  encoder = skipthoughts('uni')
  write(root + 'building train')
  train = encoder.encode(dtrain, use_norm=False, verbose=False, batch_size=32)
  write(root + 'building test ')
  test = encoder.encode(dtest, use_norm=False, verbose=False, batch_size=32)
  write(root + 'cross-validating')
  clf = cv_and_fit(train, ltrain, fit_intercept=False)
  write(root + str(np.round(100*clf.score(test, np.array(ltest)), 2))[:5] + 13*' ' + '\n')

  root = '\t\rSkipThoughts (bi):\t'
  encoder = skipthoughts('bi')
  write(root + 'building train')
  train = np.hstack([train, encoder.encode(dtrain, use_norm=False, verbose=False, batch_size=32)])
  write(root + 'building test ')
  test = np.hstack([test, encoder.encode(dtest, use_norm=False, verbose=False, batch_size=32)])
  write(root + 'cross-validating')
  clf = cv_and_fit(train, ltrain, fit_intercept=False)
  write(root + str(np.round(100*clf.score(test, np.array(ltest)), 2))[:5] + 13*' ' + '\n')
