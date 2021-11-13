from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import errno

def generate_3d_movable_graph(x_coord, y_coord, z_coord,
                              color_gt, color_cluster, opacity=0.6,
                              color_continuous_scale='bluered'):
    """This function generates 2 graphs using Plotly. Both graphs can be interacted with.
    The first graph colors the points using the ground truth labels, and the second
    one colors it using the clustering labels. Both graphs are saved on disk.

    :param x_coord:                 X coordinates of the encoder output
    :param y_coord:                 Y coordinates of the encoder output
    :param z_coord:                 Z coordinates of the encoder output
    :param color_gt:                Labels/Colors from Ground Truth
    :param color_cluster:           Labels/Colors from the clustering algorithm
    :param opacity:                 Opacity of the plotted coordinates. Default is 0.6
    :param color_continuous_scale:  Color scale to use for plotting
    """
    # Setting file names
    gt_filename = '3D_graph_movable_GroundTruth.html'
    clustering_filename = '3D_graph_movable_clustering.html'

    # Creating graph using Ground Truth labels
    movable_3d_figure_gt = px.scatter_3d(
        x=x_coord,
        y=y_coord,
        z=z_coord,
        color=color_gt,
        opacity=opacity,
        color_continuous_scale=color_continuous_scale
    )

    movable_3d_figure_gt.write_html(gt_filename)
    del gt_filename, movable_3d_figure_gt
# ----------------------------------------------------------------------------------------
    movable_3d_figure_cluster = px.scatter_3d(
        x=x_coord,
        y=y_coord,
        z=z_coord,
        color=color_cluster,
        opacity=opacity,
        color_continuous_scale=color_continuous_scale
    )

    movable_3d_figure_cluster.write_html(clustering_filename)
    del clustering_filename, movable_3d_figure_cluster

    # logger.info('3D movable graph generated')


def dbscan_analysis(prediction_output, eps=0.5, min_samples=4,):

    dbscan_clustering = DBSCAN(
        eps=eps, min_samples=min_samples).fit(prediction_output)

    return dbscan_clustering.labels_

# Change folder and create directory
home = '/home/x397j446/main-project/RUNS/12112021-204630_HPC_RUN_binary_crossentropy_N_3_Epochs_50_batch_32_scaled_0-1_all'
clustering = f'{home}/clustering'
os.chdir(f'{home}')
try:
    os.mkdir(f'{clustering}')
except OSError as e:
        if e.errno != errno.EEXIST:
            print(f'Directory {clustering} already exists')
finally:
    os.chdir(f'{clustering}')

# Read file
print("Reading prediction file")
predicted_data_df = pd.read_csv(f'{home}/predicted_data.csv')
print('File has been read')

# Convert file to numpy
predicted_data = predicted_data_df.to_numpy()
print("predicted data has been converted to numpy array")

# Run DBScan
db_scan_labels = dbscan_analysis(predicted_data, eps=1, min_samples=15)
print("{0} is the number of clusters".format(len(set(db_scan_labels))))
print(set(db_scan_labels))

# Extract x, y, and z coordinates
x_coord = []
y_coord = []
z_coord = []

for i in range(0, len(predicted_data)):
    x_coord.append(predicted_data[i][0])
    y_coord.append(predicted_data[i][1])
    z_coord.append(predicted_data[i][2])


gt_labels = predicted_data_df['y_test'].tolist()

generate_3d_movable_graph(x_coord, y_coord, z_coord,
                          color_gt=gt_labels, color_cluster=db_scan_labels,
                          opacity=0.6, color_continuous_scale='bluered')

predicted_data_df['clustering_label'] = db_scan_labels

predicted_data_df.to_csv(
    'master_project_final_run_dbscan_clustering.csv', index=False)
