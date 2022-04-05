import os
from typing import List

import cv2
from .dataset import Dataset

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
            
            if image is not None:
                dataset.add(image, label)
        
    return dataset