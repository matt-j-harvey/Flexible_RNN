import torch
import numpy as np
from tqdm import tqdm
import os

def train_epoch_batched(model, input_matrix, output_matrix, crtierion, optimiser, n_samples, hidden_state, batch_size=100):

    n_batches = int(np.divide(n_samples, batch_size))
    loss_list = []

    for batch_index in range(n_batches):

        # Get Input Batch
        batch_start = batch_index * batch_size
        batch_stop = batch_start + batch_size
        batch_input = input_matrix[batch_start:batch_stop]
        batch_output = output_matrix[batch_start:batch_stop]

        # Clear Gradients
        optimiser.zero_grad()

        # Get Model Prediction
        batch_prediction, neural_activity_tensor, hidden_state = model(batch_input, hidden_state.detach())

        # Get Loss
        estimation_loss = crtierion(batch_prediction, batch_output)

        # Get Gradients
        estimation_loss.backward()

        # Clip Gradients
        clipping_value = 200  # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Update Weights
        optimiser.step()

        loss_list.append(estimation_loss.detach().numpy())



    mean_loss = np.mean(loss_list)
    return mean_loss


def train_epoch(model, input_matrix, output_matrix, crtierion, optimiser, n_samples):

    # Clear Gradients
    optimiser.zero_grad()

    # Get Model Prediction
    prediction, neural_activity_tensor = model(input_matrix)

    # Get Loss
    estimation_loss = crtierion(prediction, output_matrix)

    # Get Gradients
    estimation_loss.backward()

    # Update Weights
    optimiser.step()

    mean_loss = estimation_loss.detach().numpy()

    return mean_loss



def fit_model(model, input_matrix, output_matrix, device, n_neurons, save_directory):

    # Set Training Parameters
    criterion = torch.nn.MSELoss()
    learning_rate = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_samples = np.shape(input_matrix)[0]

    # Convert Matricies To Tensors
    input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
    output_matrix = torch.tensor(output_matrix, dtype=torch.float32)

    # Training Loop
    epoch = 1
    print_step = 10

    #loss_tolerance =
    for x in tqdm(range(5000)):

        # Train Network
        hidden_state = torch.tensor(np.random.uniform(low=-0.01, high=0.01, size=(n_neurons)), dtype=torch.float, device=device)
        estimation_loss = train_epoch_batched(model, input_matrix, output_matrix, criterion, optimiser, n_samples, hidden_state)

        #estimation_loss = train_epoch(model, input_matrix, output_matrix, criterion, optimiser, n_samples)
        print("Epoch:", str(epoch).zfill(5),
              "estimation_loss:", np.around(estimation_loss, 6))

        if epoch % print_step == 0:
            torch.save(model.state_dict(), os.path.join(save_directory, 'model.pth'))

        epoch += 1