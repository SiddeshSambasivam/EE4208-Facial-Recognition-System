import os
import logging
from typing import List

import cv2
from .dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_dataset_paths(path:str) -> List[str]:
    """
    Prepares a dict of images paths for each person in the given path
    """

    persons = os.listdir(path)
    person_to_images = {}

    for p in persons:
        images_path = f"{path}/{p}"
        user_faces:List[str] = os.listdir(images_path)
        images = []
        for face in user_faces:
            images.append(f"{images_path}/{face}")

        person_to_images[p] = images 
    
    return person_to_images

def make_dataset(path:str) -> Dataset:
    """
    Make dataset from the given path.
    """

    dataset_paths = get_dataset_paths(path)
    dataset = Dataset()
	
    for label, image_paths in dataset_paths.items():

        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[0] < 300 or image.shape[1] < 300:
                logger.warning(f"Image {image_path} is too small: {image.shape}")
            image = cv2.resize(image, (300,300))
            
            if image is not None:
                dataset.add(image, label)

    dataset.encode_labels()

    return dataset