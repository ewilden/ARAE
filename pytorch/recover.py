import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from sklearn.linear_model import Lasso


# NOTE: Ported to Python from L1-Magic
def pdipm4bp(A, b, ATinvAAT=None, tol=1E-4, niter=100, biter=32):
  '''Primal-Dual Interior Point Method for solving Basis Pursuit
  Args:
    A: design matrix of size (d, n)
    b: measurement vector of length d
    ATinvAAT: precomputed matrix A^T(AA^T)^(-1); will be computed if None
    tol: solver tolerance
    niter: maximum number of interior point steps
    biter: maximum number of steps in backtracking line search
  Returns:
    sparse vector of length n
  '''

  if ATinvAAT is None:
    ATinvAAT = np.dot(A.T, inv(np.dot(A, A.T)))
  d, n = A.shape
  e = np.ones(n)
  alpha = 0.01
  beta = 0.5
  mu = 10
  gradf0 = np.hstack([np.zeros(n), e])

  x = np.dot(ATinvAAT, b)
  absx = np.abs(x)
  u = 0.95*absx + 0.1*max(absx)

  fu1 = x - u
  fu2 = -x - u
  lamu1 = -1 / fu1
  lamu2 = -1 / fu2
  v = np.dot(A, lamu2-lamu1)
  Atv = np.dot(A.T, v)
  rpri = np.dot(A, x) - b

  sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
  tau = 2*n*mu/sdg

  rcent = np.hstack([-lamu1*fu1, -lamu2*fu2]) - 1/tau
  rdual = gradf0 + np.hstack([lamu1-lamu2+Atv, -lamu1-lamu2])
  resnorm = np.sqrt(norm(rdual)**2+norm(rcent)**2+norm(rpri)**2)

  rdp = np.empty(2*n)
  rcp = np.empty(2*n)

  for i in range(niter):

    w1 = -1/tau*(1/fu2 - 1/fu1) - Atv
    w2 = -1-1/tau*(1/fu1+1/fu2)
    w3 = -rpri

    sig1 = -lamu1 / fu1 - lamu2 / fu2
    sig2 = lamu1 / fu1 - lamu2 / fu2
    sigx = sig1 - sig2**2/sig1
    if min(np.abs(sigx)) == 0.0:
      return x

    w1p = -(w3 - np.dot(A, (w1 / sigx - w2 * sig2/(sigx*sig1))))
    H11p = np.dot(A, A.T * (e/sigx)[:, np.newaxis])
    dv = np.dot(inv(H11p), w1p)
    dx = (w1 - w2*sig2/sig1 - np.dot(A.T, dv)) / sigx
    Adx = np.dot(A, dx)
    Atdv = np.dot(A.T, dv)

    du = (w2 - sig2*dx) / sig1
    dlamu1 = (lamu1 / fu1) * (du-dx) - lamu1 - 1/fu1/tau
    dlamu2 = (lamu2 / fu2) * (dx+du) - lamu2 - 1/fu2/tau

    s = 1
    indp = np.less(dlamu1, 0)
    indn = np.less(dlamu2, 0)
    if np.any(indp):
      s = min(s, min(-lamu1[indp] / dlamu1[indp]))
    if np.any(indn):
      s = min(s, min(-lamu2[indn] / dlamu2[indn]))
    indp = np.greater(dx-du, 0)
    indn = np.greater(-dx-du, 0)
    if np.any(indp):
      s = min(s, min(-fu1[indp] / (dx[indp]-du[indp])))
    if np.any(indn):
      s = min(s, min(-fu2[indn] / (-dx[indn]-du[indn])))
    s = 0.99 * s

    for j in range(biter):
      xp = x + s*dx
      up = u + s*du
      vp = v + s*dv
      Atvp = Atv + s*Atdv
      lamu1p = lamu1 + s*dlamu1
      lamu2p = lamu2 + s*dlamu2
      fu1p = xp - up
      fu2p = -xp - up
      rdp[:n] = lamu1p-lamu2p+Atvp
      rdp[n:] = -lamu1p-lamu2p
      rdp += gradf0
      rcp[:n] = -lamu1p*fu1p
      rcp[n:] = lamu2p*fu2p
      rcp -= 1/tau
      rpp = rpri + s*Adx
      s = beta*s
      if np.sqrt(norm(rdp)**2+norm(rcp)**2+norm(rpp)**2) <= (1-alpha*s)*resnorm:
        break
    else:
      return x

    x = xp
    nz = np.count_nonzero(x)
    lamu1 = lamu1p
    lamu2 = lamu2p
    fu1 = fu1p
    fu2 = fu2p
    sdg = -(np.inner(fu1, lamu1) + np.inner(fu2, lamu2))
    if sdg * nz / n < tol:
      return x

    u = up
    v = vp
    Atv = Atvp
    tau = 2*n*mu/sdg
    rpri = rpp
    rcent[:n] = lamu1*fu1
    rcent[n:] = lamu2*fu2
    rcent -= 1/tau
    rdual[:n] = lamu1-lamu2+Atv
    rdual[n:] = -lamu1+lamu2
    rdual += gradf0
    resnorm = np.sqrt(norm(rdual)**2+norm(rcent)**2+norm(rpri)**2)

  return x


# NOTE: LASSO with default noise parameter (alpha=1.0) recovers poorly. Highly recommended to use alpha <= 1E-3.
def recover_features(A, B, method='BP', threshold=None, **kwargs):
  '''recovers sparse feature signals from measurements given the design matrix
  Args:
    A: design matrix of size (d, V)
    B: matrix of measurements of size (n_samples, d)
    method: recovery algorithm to use; must be 'BP' or 'LASSO'
    threshold: consider only columns of A whose dot product with measurement vector is >= threshold; uses all columns if None
    kwargs: kwargs to pass to solvers; for LASSO 'fit_intercept' and 'positive' will be set automatically
  Returns:
    matrix (in CSR format) of recovered signals of size (n_samples, V)
  '''

  _, V = A.shape
  n_samples, _ = B.shape
  output = sp.lil_matrix((n_samples, V))

  if method == 'LASSO':
    kwargs['fit_intercept'] = False
    kwargs['positive'] = True
  elif not method == 'BP':
    raise(NotImplementedError)

  if threshold is None:
    if method == 'BP':
      ATinvAAT = np.dot(A.T, inv(np.dot(A, A.T)))
      for i, b in enumerate(B):
        output[i] = np.maximum(np.round(pdipm4bp(A, b, ATinvAAT=ATinvAAT, **kwargs)), 0.0)
    else:
      return Lasso(**kwargs).fit(A, B.T).sparse_coef_.rint()

  else:
    if method == 'BP':
      solve = lambda X, y: np.maximum(np.round(pdipm4bp(X, y, **kwargs)), 0.0)
    else:
      solve = lambda X, y: np.round(Lasso(**kwargs).fit(X, y).coef_)

    for i, b in enumerate(B):
      above = np.dot(A.T, b) >= threshold
      output[i,above] = solve(A[:,above], b)

  return output.tocsr()


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def precision(predicted, truth):
  '''computes precision of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the precision of each sample
  '''

  predicted = sp.csr_matrix(predicted)
  truth = sp.csr_matrix(truth)

  pslice = np.array(np.greater(predicted.sum(1), 0))[:,0]
  precision = np.zeros(pslice.shape)
  if sum(pslice):
    diff = predicted[pslice] - truth[pslice]
    precision[pslice] = np.array(1 - diff.multiply(diff>0).sum(1) / predicted[pslice].sum(1))[:,0]
  return precision


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def recall(predicted, truth):
  '''computes recall of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the recall of each sample
  '''

  predicted = sp.csr_matrix(predicted)
  truth = sp.csr_matrix(truth)

  pslice = np.array(np.greater(predicted.sum(1), 0))[:,0]
  recall = np.zeros(pslice.shape)
  if sum(pslice):
    diff = truth[pslice] - predicted[pslice]
    recall[pslice] = np.array(1 - diff.multiply(diff>0).sum(1) / truth[pslice].sum(1))[:,0]
  return recall


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def f1score(predicted, truth):
  '''computes F1-score of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the F1-score of each sample
  '''

  pr = precision(predicted, truth)
  re = recall(predicted, truth)
  pslice = np.greater(np.greater(pr, 0) + np.greater(re, 0), 0)
  f1 = np.zeros(pslice.shape)
  if sum(pslice):
    f1[pslice] = 2*pr[pslice]*re[pslice]/(pr[pslice]+re[pslice])
  return f1

import __init__
from embed import *
from load import *

def main():

  docs, _ = sst('train')
  documents = tokenize(docs)
  w2v = vocab2vecs({word for document in documents for word in document})
  dimension = list(w2v.values())[0].shape[0]
  vocab = set(w2v.keys())
  documents = [document for document in documents if vocab.issuperset(document)]
  vocab = {(word,): i for i, word in enumerate(sorted(vocab))}
  V = len(vocab)

  size = 40
  np.random.seed(0)
  subsample = set(np.random.choice(len(documents), size, replace=False))
  documents = [document for i, document in enumerate(documents) if i in subsample]
  bow = docs2bongs(documents, vocabulary=vocab)

  write('\tSST ('+str(size)+')\tPr.\tRe.\tF1\n')
  A = vocab2mat(vocab)
  B = bow.dot(A)
  X = recover_features(A.T, B, method='LASSO', alpha=1E-3)
  write('\t'.join(['', 'GloVe,  d=300:'] + [str(round(100*np.mean(func(X, bow)), 2))[:5] for func in (precision, recall, f1score)]) + '\n')

  A = vocab2mat(vocab, random='Rademacher')
  B = bow.dot(A)
  X = recover_features(A.T, B, method='LASSO', alpha=1E-3)
  write('\t'.join(['', 'Random, d=300:'] + [str(round(100*np.mean(func(X, bow)), 2))[:5] for func in (precision, recall, f1score)]) + '\n')

if __name__ == '__main__':
    main()
