


# A python code example of Deep Neural Network (with two hidden layers) from scratch without using any library except numpy. to train on a simple data set of Fibonacci series.


import numpy as np



# Print floats in readable format to print like float: 3.0, or float: 12.6666666666.
np.set_printoptions(formatter={'float': lambda x: 'float: ' + str(x)})


# This code is a definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with.

def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))


# The following code creates the input matrix, our network consists of two input nodes and one output node

X = np.array([[0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])

# The output of the Fibonacci series function as follows.
# Fibonacci series is a series of numbers in which each number ( Fibonacci number ) is the sum of the two preceding numbers. The simplest is the series 1, 1, 2, 3, 5, 8, etc.
y = np.array([[0],
              [1],
              [1],
              [1],
              [0],
              [1],
              [0],
              [0]])

# The seed for the random generator to return the same random numbers each time for being deterministic, which is very useful for debugging.
np.random.seed(1)


# Initialization of weights to random numbers. syn0 is weight matrix between input layer and first hidden layer.

# Synapses
l0Nodes = 3
l1Nodes = 4
syn0 = 2*np.random.random((l0Nodes,l1Nodes)) - 1

l2Nodes = 2
syn1 = 2*np.random.random((l1Nodes,l2Nodes)) - 1

l3Nodes = 1
syn2 = 2*np.random.random((l2Nodes,l3Nodes)) - 1

print("\n======= Deep Neural Network (with two hidden layers) =======")
print("==== Network Topology ", l0Nodes, " x ", l1Nodes, " x ", l2Nodes, " x ", l3Nodes, " ====\n")


# This is iteration training loop for network training. error decreases on each cycle of training by the slop of sigmoid function using gradient descent and back propagation.
iterations = 100000
for j in range(iterations):
    
    # Calculating forward through out the network
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))

    # Calculating error
    l3_error = y - l3
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l3_error))))
        
    # Back propagation of errors using the chain rule.
    l3_delta = l3_error*nonlin(l3, deriv=True)
    
    l2_error = l3_delta.dot(syn2.T)
    
    l2_delta = l2_error * nonlin(l2,deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # Updating weights (no alpha learning term here..)
    # Default Weight assignment equation : W = W + alpha.input.error
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("\nOutput after training ", iterations, " iterations")
print(l3)
    
    



