import time
import logging    

import cv2 
import numpy as np
from sklearn.svm import SVC

from data import Dataset
from sklearn.decomposition import PCA
# from features import PCA 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_detection_model(file_path):

    prototxt_file = file_path + '/resnet_ssd.prototxt'
    caffemodel_file = file_path + '/Res10_300x300_SSD_iter_140000.caffemodel'

    net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
    logger.info("Loaded ResNet SSD model from {}".format(file_path))    

    return net

def fit_model(dataset: Dataset, dim_size=50):    
    
    logger.info("Creating eigenfaces...")

    images = dataset.get_flatten_images()
    labels = dataset.get_labels()

    pca = PCA(
        n_components=dim_size,
        whiten=True)

    start = time.time()
    embeddings = pca.fit_transform(np.array(images))
    end = time.time()

    logger.info("Eigenfaces computation took {:.2f} seconds".format(end - start))

    model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42, degree=5)
    model.fit(embeddings,labels) 

    return model, pca
