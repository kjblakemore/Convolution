import numpy as np
from random import shuffle
from past.builtins import xrange

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
  margin_count = 0  # the number of classes which did not reach desired margin
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    margin_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        margin_count += 1
        dW.T[j] += X[i]            # accumulate gradient for incorrect classes

    dW.T[y[i]] -= margin_count * X[i]   # accumulate gradient of correct class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  print('total loss: ', loss)
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW = dW/num_train + reg * W     # average gradient + regularization gradient 

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
  
  # Accumlate the loss in the scores array
  scores = X.dot(W)

  scores = scores - np.reshape(scores[np.arange(num_train), y], (-1,1)) + 1
  scores[scores < 0] = 0
  scores[np.arange(num_train), y] = 0 # correct class does not contribute to loss
  loss = np.sum(scores)/num_train + 0.5 * reg * np.sum(W*W)

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
  # adjust scores for incorrect class gradient calculation
  scores[scores > 0] = 1
  # adjust scores for correct class gradient calculation
  scores[np.arange(num_train), y] = -np.sum(scores, axis=1) # correct class
  dW = X.T.dot(scores)

  dW = dW/num_train + reg * W     # average gradient + regularization gradient 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
