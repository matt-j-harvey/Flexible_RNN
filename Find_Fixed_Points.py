import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import scipy
from sklearn.decomposition import PCA, FactorAnalysis
from tqdm import tqdm




def find_local_minima(x, input, model):

    # Convert Initial State TO Pytorch Tensor
    x = torch.tensor(x, device='cpu', dtype=torch.float, requires_grad=True)
    x.retain_grad()

    tolerance = 0.0001
    gamma = 0.1
    gamma_decrement_window = 100

    count = 0
    max_iters = 1000

    for iteration in range(max_iters):

        # compute fn
        y = model(input, x)

        # Get Energy
        q = torch.norm(y - x)

        # Check For Convergence
        if q < tolerance:
            print("Converged! ")
            fixed_point = x.data.numpy()
            return fixed_point


        # Else Step Backwards Along Derivative
        q.backward()

        # move in direction of / opposite to grads
        x = x - (gamma * x.grad)
        x.retain_grad()

        # Update Count
        count += 1

        if count % gamma_decrement_window == 0:
            gamma *= 0.5

    print("Max iter reached")
    return x.data.numpy()



def sample_initial_states(neural_activity, n_initial_points=10):
    # Get Random Sample Of Initial States
    n_samples, n_dimensions = np.shape(neural_activity)
    states_index_list = list(range(n_samples))
    selected_state_indexes = np.random.choice(states_index_list, size=n_initial_points, replace=False)
    initial_states_list = neural_activity[selected_state_indexes]
    return initial_states_list


def find_fixed_points(input_vector, gru_cell, neural_activity):

    # Sample Initial States
    initial_states_list = sample_initial_states(neural_activity)

    # Find Fixed Points
    fixed_point_list = []
    for initial_state in tqdm(initial_states_list):
        result = find_local_minima(initial_state, input_vector, gru_cell)
        if not np.isnan(np.sum(result)):
            fixed_point_list.append(result)

    fixed_point_list = np.array(fixed_point_list)
    print("Fixed Points List", np.shape(fixed_point_list))
    return fixed_point_list

