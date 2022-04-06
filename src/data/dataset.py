from typing import List
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder

@dataclass
class FacialData:
    """
    Facial data class.
    """
    image: np.ndarray 
    label: str

class Dataset:
    """
    Abstracts the dataset management ops
    """

    def __init__(self):
        self.data: List[FacialData] = []
        self.encoder = LabelEncoder() 
    
    def add(self, image: np.ndarray, label: str):
        self.data.append(FacialData(image, label))

    def get_labels(self) -> List[str]:
        return [self.data[i].label for i in range(len(self.data))]

    def get_images(self) -> List[np.ndarray]:
        return [self.data[i].image for i in range(len(self.data))]

    def encode_labels(self) -> None:
        """
        Encode the labels to integers.
        """
        self.encoder.fit(self.get_labels())
        for i in range(len(self.data)):
            self.data[i].label = self.encoder.transform([self.data[i].label])[0]
    
    def get_flatten_images(self) -> List[np.ndarray]:
        """
        Flatten the images to a single dimension.
        """
        return [image.flatten() for image in self.get_images()]

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"Dataset<size={len(self.data)}>"
