import pandas import read_csv
from numpy import zeros, ones, dot, sum as npSum, hstack, divide, log, argmax, max as npMax, flip, array
def forward(V, a, b, initial_distribution):
    alpha = zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    return alpha
def backward(V, a, b):
    beta = zeros((V.shape[0], a.shape[0]))
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = ones((a.shape[0]))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
    return beta
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
        xi = zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = dot(dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        gamma = npSum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        # Add additional T'th element in gamma
        gamma = hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        K = b.shape[1]
        denominator = npSum(gamma, axis=1)
        for l in range(K):
            b[:, l] = npSum(gamma[:, V == l], axis=1)
        b = divide(b, denominator.reshape((-1, 1)))
    return (a, b)
def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
    omega = zeros((T, M))
    omega[0, :] = log(initial_distribution * b[:, V[0]])
    prev = zeros((T - 1, M))
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + log(a[:, j]) + log(b[j, V[t]])
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = argmax(probability)
            # This is the probability of the most probable state (2)
            omega[t, j] = npMax(probability)
    # Path Array
    S = zeros(T)
    # Find the most probable last hidden state
    last_state = argmax(omega[T - 1, :])
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    # Flip the path array since we were backtracking
    S = flip(S, axis=0)
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")
    return result
data = read_csv('data_python.csv')
V = data['Visible'].values
# Transition Probabilities
a = ones((2, 2))
a = a / npSum(a, axis=1)
# Emission Probabilities
b = array(((1, 3, 5), (2, 4, 6)))
b = b / npSum(b, axis=1).reshape((-1, 1))
# Equal Probabilities for the initial distribution
initial_distribution = array((0.5, 0.5))
a, b = baum_welch(V, a, b, initial_distribution, n_iter=100)
print(viterbi(V, a, b, initial_distribution))
