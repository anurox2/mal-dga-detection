# -*- coding: utf-8 -*-
import os

from tensorflow.keras import losses

from utilities.file_utils import get_cwd_filename, read_data
from utilities.data_processing import process_data, data_scaling
from utilities.autoencoder_design import design, get_neurons
from utilities.plotting import plotcolor, generate_2d_plot, generate_3d_plot, generate_training_plots, generate_3d_movable_graph
from prediction_data_processing import save_pred_data
from utilities.confusion_matrix import calculate_and_generate_confusion_matrix, calculate_and_generate_difference_file
from classification import kmeans_analyis, dbscan_analysis, generate_graphs
from utilities.plotting import generate_training_plots

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler

# Setting up logging
file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# Reading the ORIGINAL data file
# randomized_data_df = read_data('data/randomized_doh_data.csv')

# Reading the new data files
data_dict = {
    'all': 'data/data_for_testing/all_data.csv',
    'garrett': 'data/data_for_testing/g_data.csv',
    'no_mal': 'data/data_for_testing/no_mal_data.csv',
    'original': 'data/data_for_testing/original_data.csv',
    'mal_data': 'data/data_for_testing/mal_data.csv',
}


# Reset home directory
# os.chdir('/home/x397j446/main-project/')
data_file_name = 'all'

randomized_data_df = read_data(data_dict[data_file_name])


# Setting up variables for run
epochs = 50
batch_size = 32
scaled = True
scale_range = (0, 1)

# Subset data
subset_data = False
if(subset_data):
    randomized_data_df = randomized_data_df.head(10000)

try:
    randomized_data_df.drop(columns=['datasrc'], inplace=True)
except Exception as e:
    logger.error('Error occured. Column "datasrc" not found')
    logger.exception(e)


# activation = 'linear'
activation = 'softmax'

# Defining design structures
design_number = 17  # 7  11  12
design_structures = {
    # 24 inputs --------------------- 3 outputs
    1: '24_8_3',
    2: '24_12_6_3',  # --- scenario 1
    3: '24_18_12_8_6_3',
    4: '24_20_16_12_8_6_3',
    5: '24_20_16_12_8_4_3',
    6: '24_12_3',

    # 24 inputs --------------------- 4 outputs
    7: '24_20_16_12_8_4',
    8: '24_8_4',

    # 15 inputs --------------------- 3 outputs
    9: '15_7_8_6_3',

    # 76 inputs ---------------------
    10: '76_38_19_10_8_6_3',

    # 23 inputs --------------------- 3 outputs
    11: '23_20_17_14_11_8_5_3',
    12: '23_20_17_14_11_8_6_4_3',
    13: '23_12_6_3',
    14: '23_22_11_10_5_3',  # --- scenario 2 - SUCCESS FULL ON ORIGINAL DATA
    15: '23_11_5_3',
    16: '23_12_6_4',

    # 17 inputs ---- Mix of old data + Garrett's Data + Sergio's Data
    # The OLD data was not extracted from PCAPs again.
    17: '17_16_8_4_3',

}


# Fetching neurons and autoencoder design structure
design_structure = design_structures[design_number]
neurons = get_neurons(design_structure)

# Setting loss function to use
loss_function_dict = {
    1: 'mse',
    2: losses.mean_squared_logarithmic_error,
    3: 'binary_crossentropy',
}
loss_function_number = 3
loss_function = loss_function_dict[loss_function_number]


clustering_method_selection = 2
cluster_dict = {
    1: 'dbscan',
    2: 'kmeans',
    3: 'optics',
}
clustering_method = cluster_dict[clustering_method_selection]


# Variables for classification
kmeans_clusters = 2
ranmdom_state = None

# Starting Execution
logger.info(
    '---------------------------------------------------------------------------------------------')
logger.info('''STARTING EXECUTION
                                            Loss function - {0}
                                            Neurons - {1}
                                            Scaled - {2}
                                            Scale Range - {3}
                                            Batch Size  - {4}
                                            Epochs - {5}'''.format(
    str(loss_function),
    str(neurons),
    str(scaled),
    str(scale_range),
    str(batch_size),
    str(epochs),
))

# Extract folder name
dir_name = get_cwd_filename(
    loss_function,
    neurons,
    scaled,
    scale_range,
    batch_size,
    epochs,
    data_file_name,
)

# Creating new folder for the run
try:
    os.mkdir(dir_name)
except Exception:
    logger.error(
        'EXITING CODE. Could not create directory to store deliverables')
    logger.exception(Exception)
    print('EXITING CODE. Could not create directory to store deliverables')
    exit(100)

os.chdir(dir_name)
logger.info('Directory changed to {0}'.format(os.getcwd()))


# -------------------------------------------------------------------------------------------------------
# Processing data
X, y, X_train, X_test, y_train, y_test = process_data(randomized_data_df, data_file_name)
print("\n")
print("X shape", X.shape[1])
print("\n")

# Scaling data based on option
if(scaled):
    X_train, X_test = data_scaling(X, X_train, X_test, scale_range)

# Autoencoder model design
logger.info(
    "Starting autoencoder process for loss function {0}".format(loss_function))
model, encoder_no_decoder = design(
    design_structure, X, loss_function, activation=activation)
logger.info("Autoencoder and encoder models built")

# Training
logger.info("Training process started")
history = model.fit(X_train,
                    X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, X_test)
                    )
logger.info("Training process completed")

# Plotting results
try:
    # logger.info("The accuracy is", history.history['accuracy'])
    print("The accuracy is", history.history['accuracy'])
except Exception:
    logger.exception(
        'Accuracy could NOT be extracted. Exception:{0}'.format(Exception))
    print('Accuracy could NOT be extracted. Check log file for more details')

# Printing accuracy to file.
accuracy_str = "The accuracy is" + \
    str(history.history['accuracy']) + "\nThe value accuracy is" + \
    str(history.history['val_accuracy'])
acc_text_file = open("accuracy.txt", "w")
n = acc_text_file.write(accuracy_str)
acc_text_file.close()

generate_training_plots(history)

# PREDICTION
prediction_output = encoder_no_decoder.predict(X_test)
logger.info('Prediction completed. Predicted and ACTUAL values in console')
# print("######## PREDICTION values ########")
# print(prediction_output)

# print("######## ACTUAL values ############")
# print(y_test)

# Get colors for ground truth
plt_colors = plotcolor(y_test)

save_pred_data(neurons, prediction_output, y_test)

# Extracting x, y and z coordinates for plotting.
x_coord = []
y_coord = []
z_coord = []
a_coord = []
for i in range(0, len(prediction_output)):
    x_coord.append(prediction_output[i][0])
    y_coord.append(prediction_output[i][1])
    if(neurons == 3):
        z_coord.append(prediction_output[i][2])
    if(neurons == 4):
        z_coord.append(prediction_output[i][2])
        a_coord.append(prediction_output[i][3])

generate_3d_movable_graph(x_coord, y_coord, z_coord, plt_colors,
                        0.6, 'bluered', False)


if(neurons == 2):
    logger.info('Printing 2D graph')
    generate_2d_plot(x_coord, y_coord, plt_colors)
    logger.info('Graph printed')

elif(neurons == 3):
    z_coord = []
    for i in range(0, len(prediction_output)):
        z_coord.append(prediction_output[i][2])
    logger.info('Printing 3D graph')
    generate_3d_plot(x_coord, y_coord, z_coord, plt_colors)
    logger.info('Graph printed')

elif(neurons == 4):
    z_coord = []
    a_coord = []
    for i in range(0, len(prediction_output)):
        z_coord.append(prediction_output[i][2])
        a_coord.append(prediction_output[i][2])
    generate_3d_plot(x_coord, y_coord, z_coord, plt_colors, False, 'xyz')
    generate_3d_plot(x_coord, y_coord, a_coord, plt_colors, False, 'xya')
    generate_3d_plot(a_coord, y_coord, z_coord, plt_colors, False, 'ayz')
    generate_3d_plot(x_coord, z_coord, a_coord, plt_colors, False, 'xza')

else:
    logger.warn(
        'Too many neurons. Saving prediction output for further analysis')
    print('Too many neurons. Saving prediction output for further analysis')


# if(clustering_method == 'dbscan'):
#     clustering_labels = dbscan_analysis(
#         prediction_output, eps=0.5, min_samples=5)

# elif(clustering_method == 'kmeans'):
#     clustering_labels = kmeans_analyis(
#         prediction_output,
#         init_funnction='k-means++',
#         n_clusters=kmeans_clusters,
#         n_init=10,
#         max_iter=300,
#         algorithm='elkan',
#         precompute_distances=True,
#         random_state=None,
#     )
# else:
#     logger.error('Incorrect clustering method selected')

# print("\n\n\nCLUSTERING LABELS: {0}\n\n\n".format(set(clustering_labels)))
# save_pred_data(neurons, prediction_output, y_test, clustering_labels)

# if(z_coord is not None and a_coord is None):
#     generate_3d_movable_graph(
#         x_coord, y_coord, z_coord, clustering_labels, 0.6, 'bluered', True, clustering_method)
#     generate_3d_movable_graph(
#         x_coord, y_coord, z_coord, y_test, 0.6, 'bluered', False)
# elif(z_coord is not None and a_coord is not None):
#     # Clustering labeled graphs
#     logger.info('Creating 3d movable graphs with clustering labels')
#     # XYZ graph
#     generate_3d_movable_graph(
#         x_coord, y_coord, z_coord, clustering_labels, 0.6,
#         'bluered', True, clustering_method, 'xyz')
#     # XYA graph
#     generate_3d_movable_graph(
#         x_coord, y_coord, a_coord, clustering_labels, 0.6,
#         'bluered', True, clustering_method, 'xya')
#     # AYZ graph
#     generate_3d_movable_graph(
#         a_coord, y_coord, z_coord, clustering_labels, 0.6,
#         'bluered', True, clustering_method, 'ayz')
#     # XZA graph
#     generate_3d_movable_graph(
#         x_coord, z_coord, a_coord, clustering_labels, 0.6,
#         'bluered', True, clustering_method, 'xza')

#     # Ground Truth graphs
#     logger.info('Creating 3d movable graphs with Ground Truth labels')
#     # XYZ graph
#     generate_3d_movable_graph(
#         x_coord, y_coord, z_coord, y_test, 0.6, 'bluered', False, None, 'xyz')
#     # XYA graph
#     generate_3d_movable_graph(
#         x_coord, y_coord, a_coord, y_test, 0.6, 'bluered', False, None, 'xya')
#     # AYZ graph
#     generate_3d_movable_graph(
#         a_coord, y_coord, z_coord, y_test, 0.6, 'bluered', False, None, 'ayz')
#     # XZA graph
#     generate_3d_movable_graph(
#         x_coord, z_coord, a_coord, y_test, 0.6, 'bluered', False, None, 'xza')
# else:
#     logger.error('Unknown number of neurons')


# calculate_and_generate_confusion_matrix(y_test, clustering_labels)
# calculate_and_generate_difference_file(y_test, clustering_labels)
