# -------------------------------------------------------- #
# -------------------------------------------------------- #
# implementing a graph neural network with an
# attention mechanism from scratch for node classification
# based on node attributes
# -------------------------------------------------------- #
# -------------------------------------------------------- #

import numpy as np
import torch

# Adjacency matrix
A = torch.tensor([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], dtype=torch.float, requires_grad=True)

# Node attributes
X = torch.tensor([[0.8, 0],
              [0, 1],
              [0.05, 0],
              [0, 0.05]], dtype=torch.float)

W1 = torch.tensor([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6],
               [0.7, 0.8]], dtype=torch.float, requires_grad=True)

b1 = torch.tensor([0.1, 0.2,0.3,0.4], dtype=torch.float, requires_grad=True)

W2 = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float, requires_grad=True)

b2 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float, requires_grad=True)

target = torch.tensor([1, 1, 0, 0], dtype=torch.long)

learning_rate = 0.05


def compute_attention_weights(A, X, W1, b1, W2, b2):
  # Compute hidden representation
  dot_product = torch.matmul(X, W1.T)
  # Sum the dot product along the second dimension (axis=1)
  sum_ = torch.sum(dot_product, dim=1)
  # Add b1
  sum_ = torch.add(sum_, b1)
  # Apply the hyperbolic tangent function
  H = torch.tanh(sum_)
  # Reshape H
  H = H.view(-1, 1)

  # Compute attention weights
  # Compute the dot product of H and W2
  dot_product = torch.matmul(H, W2)
  # Sum the dot product along the second dimension (axis=1)
  sum_ = torch.sum(dot_product, dim=1)
  # Add b2
  sum_ = torch.add(sum_, b2)
  # Compute the exponential of the result
  attention_weights = torch.exp(sum_)


  # Normalize attention weights
  # Sum attention_weights
  sum_attention_weights = torch.sum(attention_weights)
  # Divide attention_weights by the sum
  attention_weights = torch.div(attention_weights, sum_attention_weights)

  # Update adjacency matrix with attention weights
  A = A * attention_weights

  return A


for epoch in range(1000):
  A = compute_attention_weights(A, X, W1, b1, W2, b2)

  # Weighted node attributes
  X_weighted = A @ X

  # Hidden representation (using weighted node attributes as input)
  # Compute the dot product of X_weighted and the transpose of W1
  dot_product = torch.matmul(X_weighted, W1.T)
  # Sum the dot product along the second dimension (axis=1)
  sum_ = torch.sum(dot_product, dim=1)
  # Add b1
  sum_ = sum_ + b1
  # Apply the tanh function
  H = torch.tanh(sum_)
  # Reshape H
  H = H.view(-1, 1)

  # Output
  # Compute the dot product of H and W2
  dot_product = torch.matmul(H, W2)
  # Sum the dot product along the second dimension (axis=1)
  sum_ = torch.sum(dot_product, dim=1)
  # Add b2
  sum_ = sum_ + b2
  # Compute the log of the result
  output = torch.log(sum_)

  # Loss
  ##target = torch.tensor([1, 1, 0, 0], dtype=torch.float)
  ##loss = target-output
  ##loss = loss.sum()

  # Output
  logits = output
  # Cross-entropy loss

  # Compute log-softmax along the first dimension (dim=0)
  log_softmax = torch.nn.functional.log_softmax(logits, dim=0)
  # Compute negative log-likelihood loss
  #loss = torch.nn.functional.nll_loss(log_softmax, target)
  loss = torch.nn.functional.smooth_l1_loss(output, target)

  # Classification logits (using hidden representation as input)
  logits = output

  # Predicted class label
  # Compute the exponential of the negative of logits
  exp = torch.exp(-logits)
  # Add 1 to the exponential
  exp = torch.add(exp, 1)
  # Divide 1 by the result
  probs = torch.div(1, exp)

  prediction = torch.argmax(logits)

  #loss.backward()

  regularization_lambda = 0.01
  regularization_loss = torch.sum(W1 ** 2) + torch.sum(W2 ** 2)
  loss += regularization_lambda * regularization_loss

  loss.backward(retain_graph=True)

  # Update parameters using gradient descent

  # Do not track gradients while updating the parameters
  with torch.no_grad():
    # Update W1
    W1_new = W1 - learning_rate * W1.grad
    W1.data.copy_(W1_new)
    W1.grad.zero_()

    # Update b1
    b1_new = b1 - learning_rate * b1.grad
    b1.data.copy_(b1_new)
    b1.grad.zero_()

    # Update W2
    W2_new = W2 - learning_rate * W2.grad
    W2.data.copy_(W2_new)
    W2.grad.zero_()

    # Update b2
    b2_new = b2 - learning_rate * b2.grad
    b2.data.copy_(b2_new)
    b2.grad.zero_()

    #print(W1.grad)
    #print(W1)

    # Print the loss
  print(f"Epoch: {epoch + 1}, Loss: {loss}")


print(output)





