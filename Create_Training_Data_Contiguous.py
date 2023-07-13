import numpy as np


def create_visual_trial():

    trial_input_matrix = np.zeros((80, 4))
    trial_output_matrix = np.zeros((80, 2))
    visual_stim = np.random.randint(low=0, high=2)

    if visual_stim == 0:
        trial_input_matrix[60:80, 0] = 1

    elif visual_stim == 1:
        trial_input_matrix[60:80, 1] = 1

    trial_output_matrix[:, 0] = 1
    return trial_input_matrix, trial_output_matrix


def create_odour_trial():

    # Create Empty Data Matrix
    trial_input_matrix = np.zeros((120, 4))
    trial_output_matrix = np.zeros((120, 2))

    # Get Irrel Status
    irrel_status = np.random.uniform(low=0, high=1)
    if irrel_status < 0.7:
        irrel_status = 1
    else:
        irrel_status = 0

    if irrel_status == 1:

        # Get Visual Stim
        visual_stim = np.random.randint(low=0, high=2)
        if visual_stim == 0:
            trial_input_matrix[60:80, 0] = 1

        elif visual_stim == 1:
            trial_input_matrix[60:80, 1] = 1

    # Get Odour Stim
    odour_stim = np.random.randint(low=0, high=2)
    if odour_stim == 0:
        trial_input_matrix[100:120, 2] = 1

    elif odour_stim == 1:
        trial_input_matrix[100:120, 3] = 1

    trial_output_matrix[:, 1] = 1
    return trial_input_matrix, trial_output_matrix


def create_training_data(n_blocks=10, min_block_size=10, max_block_size=20):

    """
    Inputs
    vis 1
    vis 2
    odour 1
    odour 2

    Outputs
    context 1
    context 2


    Visual Trials
    60 timepoints of Pre Stim
    20 timepoints of Visual Stimulus + Response

    Odour Trials
    60 timepoints of Pre Stim
    20 timepoints of Visual Stimulus
    20 Timepoints of Interval
    20 timepoints of Odour Stimulus + Response
    """

    current_block_type = 0
    input_data = []
    output_data = []
    for block_index in range(n_blocks):

        # Get Block Size
        block_size = int(np.random.uniform(low=min_block_size, high=max_block_size))

        # Iterate Through Trials
        for trial_index in range(block_size):

            if current_block_type == 0:
                trial_input_matrix, trial_output_matrix = create_visual_trial()

            elif current_block_type == 1:
                trial_input_matrix, trial_output_matrix = create_odour_trial()

            # Add Trial Data
            input_data.append(trial_input_matrix)
            output_data.append(trial_output_matrix)

        # Switch Block
        if current_block_type == 0:
            current_block_type = 1
        elif current_block_type == 1:
            current_block_type = 0

    input_data = np.vstack(input_data)
    output_data = np.vstack(output_data)
    print("Input Data", np.shape(input_data))
    print("Output Data", np.shape(output_data))

    return input_data, output_data


def plot_data(input_data, output_data):

    plt.plot(input_data[:, 0], c='b')
    plt.plot(input_data[:, 1], c='r')
    plt.plot(input_data[:, 2], c='g')
    plt.plot(input_data[:, 3], c='m')
    plt.plot(output_data[:, 0], c='orange')
    plt.show()

