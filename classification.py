from utilities.plotting import generate_3d_plot, generate_2d_plot
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler


file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def kmeans_analyis(
        prediction_output, init_funnction, n_clusters, n_init,
        max_iter, algorithm='elkan', precompute_distances=None, 
        random_state=None,):
    # Setting up KMeans object
    kmeans = KMeans(
        init=init_funnction,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        precompute_distances=precompute_distances,
        algorithm=algorithm
    )
    kmeans.fit(prediction_output)

    logger.info(
        'Inertia of predicted data based on K-Means is {0}'.format(kmeans.inertia_))
    logger.info(
        'Cluster centers predicted data based on K-Means is {0}'.format(kmeans.cluster_centers_))
    logger.info('Iterations required {0}'.format(kmeans.n_iter_))
    logger.info('KMEANS LABELS {0}'.format(kmeans.labels_))    

    return kmeans.labels_


def dbscan_analysis(prediction_output, eps=0.5, min_samples=4,):

    dbscan_clustering = DBSCAN(
        eps=eps, min_samples=min_samples).fit(prediction_output)

    return dbscan_clustering.labels_


def generate_graphs(
        x_coord, y_coord, neurons, clustering_labels,
        clustering_bool=False, z_coord=None, a_coord=None):

    if(neurons == 2):
        generate_2d_plot(x_coord, y_coord, clustering_labels, clustering_bool)
    elif(neurons == 3):
        generate_3d_plot(x_coord, y_coord, z_coord,
                         clustering_labels, clustering_bool)
    elif(neurons == 4):
        generate_3d_plot(x_coord, y_coord, z_coord,
                         clustering_labels, clustering_bool, 'xyz')
        generate_3d_plot(x_coord, y_coord, a_coord,
                         clustering_labels, clustering_bool, 'xya')
        generate_3d_plot(a_coord, y_coord, z_coord,
                         clustering_labels, clustering_bool, 'ayz')
        generate_3d_plot(x_coord, z_coord, a_coord,
                         clustering_labels, clustering_bool, 'xza')
    else:
        logger.info('KMEANS graph cannot be generated. Neuron count is above 3')
