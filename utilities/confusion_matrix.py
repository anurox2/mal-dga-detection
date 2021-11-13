from sklearn.metrics import confusion_matrix

import logging
import pandas as pd
from utilities.logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def calculate_and_generate_confusion_matrix(y_test, classification_labels):
    conf_matrix = confusion_matrix(y_test, classification_labels)
    logger.info('Confusion matrix generated')
    print(conf_matrix)

    f = open('confusion_matrix.txt', 'w')
    f.write('CONF_MATRIX\n' + repr(conf_matrix) + '\n')
    f.close()
    logger.info('Confusion matrix written to file')



def calculate_and_generate_difference_file(ground_truth, k_means_labels):
    logger.info('Creating a difference file with ground truth labels vs KMeans labels')
    df = pd.DataFrame(
        list(zip(ground_truth, k_means_labels)),
        columns =['ground_truth', 'k_means_labels']
    )
    logger.info('Difference between ground truth and KMeans labels calculated')
    df['difference'] = df['ground_truth'].astype(int) - df['k_means_labels'].astype(int)
    
    df.to_csv('difference_file.csv')
    logger.info('Difference file written to CSV')