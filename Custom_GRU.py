import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


class custom_rnn(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, n_outputs, device):
        super(custom_rnn, self).__init__()

        # Save Parameters
        self.device = device
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.gru_cell = torch.nn.GRUCell(self.n_inputs, self.n_neurons, bias=True, device=device, dtype=torch.float32)

        output_weights = np.random.uniform(low=-0.01, high=0.01, size=(self.n_neurons, self.n_outputs))
        output_biases = np.random.uniform(low=-0.01, high=0.01, size=(self.n_outputs))

        # Initialise Weights
        self.output_weights = torch.nn.Parameter(torch.tensor(output_weights, dtype=torch.float, device=device))
        self.output_biases = torch.nn.Parameter(torch.tensor(output_biases, dtype=torch.float, device=device))


    def forward(self, external_input_tensor, hidden_state):

        number_of_samples = external_input_tensor.size(dim=0)
        output_tensor = torch.zeros(size=(number_of_samples, self.n_outputs), device=self.device)
        neural_activity_tensor = torch.zeros(size=(number_of_samples, self.n_neurons), device=self.device)

        # Initialise Random Hidden State
        for sample_index in range(number_of_samples):

            # Put Through GRU
            hidden_state = self.gru_cell(external_input_tensor[sample_index], hidden_state)

            # Get Output
            output_tensor[sample_index] = torch.matmul(hidden_state, self.output_weights) + self.output_biases
            neural_activity_tensor[sample_index] = hidden_state

        return output_tensor, neural_activity_tensor, hidden_state
