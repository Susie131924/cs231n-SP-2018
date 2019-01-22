import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  num_train = X.shape[0]
  for i in range(num_train):
      f = scores[i] - np.max(scores[i])
      p = np.sum(np.exp(f))
      loss += -np.log(np.exp(f[y[i]]) / p)
      for j in range(W.shape[1]):
          dW[:, j] += X[i] * np.exp(f[j]) / p
      dW[:, y[i]] -= X[i]
  
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  
  f = scores - np.max(scores)
  p = np.sum(np.exp(f), axis=1)
  loss = np.sum(np.log(p)) - np.sum(f[range(num_train), y]) 
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)
  
  
  margin = np.exp(f) / p[:, None]
  margin[range(num_train), y] -= 1
  dW = X.T.dot(margin) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

