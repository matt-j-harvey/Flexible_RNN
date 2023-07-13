import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_latent_space_extent(transformed_data):

    # Get Extent
    min_x = np.min(transformed_data[:, 0])
    max_x = np.max(transformed_data[:, 0])
    min_y = np.min(transformed_data[:, 1])
    max_y = np.max(transformed_data[:, 1])

    x_range = np.abs(np.subtract(max_x, min_x))
    y_range = np.abs(np.subtract(max_y, min_y))

    additional_spread = 0.1
    min_x = min_x - (x_range * additional_spread)
    max_x = max_x + (x_range * additional_spread)
    min_y = min_y - (y_range * additional_spread)
    max_y = max_y + (y_range * additional_spread)

    return min_x, max_x, min_y, max_y



def get_energy_field(pca_model, gru_cell, input_vector, min_x, max_x,  min_y, max_y, density=100):

    energy_map = np.zeros((density, density))
    x_positions = np.linspace(start=min_x, stop=max_x, num=density)
    y_positions = np.linspace(start=min_y, stop=max_y, num=density)

    for x_index in range(density):
        for y_index in range(density):
            x_coord = x_positions[x_index]
            y_coord = y_positions[y_index]

            transformed_point = pca_model.inverse_transform([x_coord, y_coord])
            x = torch.tensor(transformed_point, device='cpu', dtype=torch.float)

            # compute fn
            y = gru_cell(input_vector, x)

            # Get Energy
            q = torch.norm(y - x)

            energy_map[y_index, x_index] = q.data

    energy_map = np.flip(energy_map, axis=0)
    #energy_map = np.flip(energy_map, axis=1)
    return energy_map




def get_vector_field(pca_model, input_vector, gru_cell, min_x, max_x,  min_y, max_y, density=20):

    # For Quiver Plots The Form Must Be
    # x - x position of arrow
    # y - y position of arrow
    # u - x direction of arrow
    # v - y direction of arrow

    x_list = []
    y_list = []
    u_list = []
    v_list = []

    x_positions = np.linspace(start=min_x, stop=max_x, num=density)
    y_positions = np.linspace(start=min_y, stop=max_y, num=density)

    for x_index in range(density):
        for y_index in range(density):
            x_coord = x_positions[x_index]
            y_coord = y_positions[y_index]

            transformed_point = pca_model.inverse_transform([x_coord, y_coord])

            # Get New Point
            transformed_point = torch.tensor(transformed_point, dtype=torch.float)
            new_point = gru_cell(input_vector, transformed_point).detach().numpy()

            old_point_pca = pca_model.transform(transformed_point.reshape(1, -1))[0]
            new_point_pca = pca_model.transform(new_point.reshape(1, -1))[0]

            derivative = np.subtract(new_point_pca, old_point_pca)

            x_list.append(x_coord)
            y_list.append(y_coord)
            u_list.append(derivative[0])
            v_list.append(derivative[1])

    return [x_list, y_list, u_list, v_list]


def visualise_neural_activity_3d(neural_activity, fixed_points_list=None):

    pca_model = PCA(n_components=3)
    transformed_data = pca_model.fit_transform(neural_activity)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1, projection='3d')
    axis_1.plot(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.4)

    if fixed_points_list != None:
        transformed_fixed_points = pca_model.transform(fixed_points_list)
        axis_1.scatter(transformed_fixed_points[:, 0], transformed_fixed_points[:, 1], transformed_fixed_points[:, 2], c='m')

    plt.show()

