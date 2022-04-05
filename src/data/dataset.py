from typing import List
from dataclasses import dataclass

import numpy as np

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
    
    def add(self, image: np.ndarray, label: str):
        self.data.append(FacialData(image, label))

    def get_labels(self) -> List[str]:
        return [self.data[i].label for i in range(len(self.data))]

    def get_images(self) -> List[np.ndarray]:
        return [self.data[i].image for i in range(len(self.data))]

    def get_label(self, index: int) -> str:
        return self.data[index].label

    def get_image(self, index: int) -> np.ndarray:
        return self.data[index].image

    def __len__(self):
        return len(self.data)

        