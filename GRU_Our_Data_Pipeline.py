import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import Create_Training_Data_Contiguous
import Custom_GRU
import Train_GRU
import Find_Fixed_Points
import RNN_Analysis_Tools




def visualise_temporal_trajectories(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y, input_data):


    n_timepoints = np.shape(transformed_data)[0]

    plt.ion()
    figure_1 = plt.figure()

    for timepoint_index in range(n_timepoints):

        axis_1 = figure_1.add_subplot(2, 1, 1)
        axis_1.plot(transformed_data[:, 0], transformed_data[:, 1], alpha=0.4, c='orange')
        axis_1.scatter(transformed_fixed_points[:, 0], transformed_fixed_points[:, 1], c='m')

        # Plot Vector Field
        axis_1.quiver(vector_field[0], vector_field[1], vector_field[2], vector_field[3])

        # Plot Energy Field
        log_energy_field = np.log10(energy_field)
        axis_1.imshow(log_energy_field, extent=[min_x, max_x,  min_y, max_y],  cmap='Blues_r')

        # Plot Current Timepoint
        current_timepoint = transformed_data[timepoint_index]
        axis_1.scatter([current_timepoint[0]], [current_timepoint[1]], c='g')


        # Plot Behaviour
        behaviour_axis = figure_1.add_subplot(2,1,2)
        window_size = 100
        timepoint_window_start = timepoint_index - window_size
        timepoint_window_stop = timepoint_index + window_size

        timepoint_window_start = np.clip(timepoint_window_start, a_min=0, a_max=n_timepoints)
        timepoint_window_stop = np.clip(timepoint_window_stop, a_min=0, a_max=n_timepoints)


        behaviour_axis.plot(input_data[timepoint_window_start:timepoint_window_stop, 0], c='b')
        behaviour_axis.plot(input_data[timepoint_window_start:timepoint_window_stop, 1], c='r')
        behaviour_axis.plot(input_data[timepoint_window_start:timepoint_window_stop, 2], c='g')
        behaviour_axis.plot(input_data[timepoint_window_start:timepoint_window_stop, 3], c='m')
        behaviour_axis.axvline(window_size, c='k', linestyle='dashed')

        plt.draw()
        plt.pause(0.1)
        plt.clf()






def visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(transformed_data[:, 0], transformed_data[:, 1], alpha=0.4, c='orange')
    axis_1.scatter(transformed_fixed_points[:, 0], transformed_fixed_points[:, 1], c='m')

    # Plot Vector Field
    axis_1.quiver(vector_field[0], vector_field[1], vector_field[2], vector_field[3])

    # Plot Energy Field
    log_energy_field = np.log10(energy_field)
    axis_1.imshow(log_energy_field, extent=[min_x, max_x,  min_y, max_y],  cmap='Blues_r')

    plt.show()




def plot_performance(input_data, output_data, prediction):

    plt.plot(input_data[:, 0], c='b')
    plt.plot(input_data[:, 1], c='r')
    plt.plot(input_data[:, 2], c='g')
    plt.plot(input_data[:, 3], c='m')
    plt.plot(output_data[:, 0], c='orange', alpha=0.5)
    plt.plot(output_data[:, 1], c='cyan', alpha=0.5)

    plt.plot(prediction[:, 0], c='orange', linestyle='dashed')
    plt.plot(prediction[:, 1], c='cyan', linestyle='dashed')
    plt.show()



# Set Save Directory
save_directory = r"C:\Users\matth\Documents\Flexible_RNN_Play\GRU_Our_Task_With_Noise"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Create Data
input_data, output_data = Create_Training_Data_Contiguous.create_training_data(n_blocks=10, min_block_size=10, max_block_size=20)
noise = np.random.normal(loc=0, scale=0.1, size=np.shape(input_data))
input_data = np.add(input_data, noise)

# Create RNN
n_inputs = np.shape(input_data)[1]
n_outputs = np.shape(output_data)[1]
n_neurons = 50
device = "cpu"
rnn = Custom_GRU.custom_rnn(n_inputs, n_neurons, n_outputs, device)
rnn.load_state_dict(torch.load(os.path.join(save_directory, 'model.pth')))

# Train Model
#Train_GRU.fit_model(rnn, input_data, output_data, device, n_neurons, save_directory)

# Visualise Neural Activity
with torch.no_grad():
    hidden_state = torch.tensor(np.random.uniform(low=-0.01, high=0.01, size=(n_neurons)), dtype=torch.float, device=device)
    output_prediction, neural_activity, hidden_state = rnn(torch.tensor(input_data, dtype=torch.float32), hidden_state)
    neural_activity = neural_activity.detach().numpy()
    output_prediction = output_prediction.detach().numpy()
print("Neural Activity", np.shape(neural_activity))

plot_performance(input_data, output_data, output_prediction)

# Find Fixed Points
gru_cell = rnn.gru_cell
input_vector = torch.zeros(n_inputs, dtype=torch.float)
fixed_points_list = Find_Fixed_Points.find_fixed_points(input_vector, gru_cell, neural_activity)

# Perform PCA
pca_model = PCA(n_components=2)
transformed_data = pca_model.fit_transform(neural_activity)
transformed_fixed_points = pca_model.transform(fixed_points_list)


# Get Latent Space Extent
min_x, max_x, min_y, max_y = RNN_Analysis_Tools.get_latent_space_extent(transformed_data)

# Get Energy Field
energy_field = RNN_Analysis_Tools.get_energy_field(pca_model, gru_cell, input_vector, min_x, max_x,  min_y, max_y)

# Get Recurrent Vector Field
vector_field = RNN_Analysis_Tools.get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x, min_y, max_y)

# Visualise Neural Activity
visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y)

visualise_temporal_trajectories(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y, input_data)


# Get Recurrent Vector Field
print("vis 1")
input_data = np.array([1,0,0,0])
input_vector = torch.tensor(input_data, dtype=torch.float)
vector_field = RNN_Analysis_Tools.get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x, min_y, max_y)
visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y)


# Get Recurrent Vector Field
print("vis 2")
input_data = np.array([0,1,0,0])
input_vector = torch.tensor(input_data, dtype=torch.float)
vector_field = RNN_Analysis_Tools.get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x, min_y, max_y)
visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y)


# Get Recurrent Vector Field
print("odr 1")
input_data = np.array([0,0,1,0])
input_vector = torch.tensor(input_data, dtype=torch.float)
vector_field = RNN_Analysis_Tools.get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x, min_y, max_y)
visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y)


# Get Recurrent Vector Field
print("odr 2")
input_data = np.array([0,0,0,1])
input_vector = torch.tensor(input_data, dtype=torch.float)
vector_field = RNN_Analysis_Tools.get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x, min_y, max_y)
visualise_neural_activity_pca_space(transformed_data, transformed_fixed_points, energy_field, vector_field, min_x, max_x,  min_y, max_y)

