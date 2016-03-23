import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    misses = 0  # number of classes that didn't meet desired margin
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        misses += 1
        dW[:,j] += X[i]         # accumulate partial gradients wrt w[j]
    dW[:,y[i]] -= misses * X[i] # accumulate partial gradients wrt w[y[i]]

  dW = dW/num_train + reg * W   # average gradient and add regularization

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y[:num_train]]
  
  # Calculate margins for each training sample
  margins = scores - correct_class_scores.reshape(-1,1) + 1
  
  # Adjust margins - correct class does not contibute to loss & max margin is 0
  margins[np.arange(num_train), y[:num_train]] = 0
  margins[margins<0] = 0
  
  # Total loss is sum of all sample losses
  loss = margins.sum() / num_train + 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Adjust margins to be used as mask for gradient calculation.  Each class of
  # the gradient is a sum of weighted X values.  For class j of the gradient,
  # Xs with class y[j] are weighted by the negative sum of positive margins
  # within the class.  Xs with class values other than y[j] are added to the
  # gradient, if their margin is positive.
  margins[margins > 0] = 1
  margins[np.arange(num_train),y[:num_train]] = -np.sum(margins, 1)

  dW = (X.T).dot(margins)

  dW = dW/num_train + reg * W   # average gradient and add regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
