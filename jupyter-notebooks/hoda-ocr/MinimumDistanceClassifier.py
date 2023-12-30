from scipy.spatial.distance import euclidean
from typing import List, Tuple, Any
import numpy as np
import sys

class MinDistanceClassifier:
    def __init__(self) -> None:
        self.class_domain_: List[Any]
        self.class_centers_: List[Tuple(List[float], Any)] = []

    def fit(self, X, y) -> None:
        """Calculates feature means from train dataset and stores them."""
        self.class_domain_ = np.array(list(set(y.flatten())))

        for label in self.class_domain_:
            class_mask = (y == label)
            feature_mean = np.mean(X[class_mask], axis=0)
            self.class_centers_.append((feature_mean, label))
        
    def predict(self, X) -> List[Any]:
        """Assigns to each point in the test dataset the class with the closest center."""
        y_pred = np.array([])
        for row in X:
            closest_distance = sys.maxsize
            closest_center = 0
            for val, label in self.class_centers_:
                dist = euclidean(row.flatten(), val)
                if dist < closest_distance:
                    closest_center = label
                    closest_distance = dist
            y_pred = np.append(y_pred, closest_center)

        return y_pred