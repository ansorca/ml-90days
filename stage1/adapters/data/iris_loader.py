
from sklearn.datasets import load_iris
from sklearn import model_selection as skms 
import numpy as np
from dataclasses import dataclass

@dataclass
class IrisData:
    train_features: np.ndarray
    train_targets: np.ndarray
    testing_features: np.ndarray
    testing_targets: np.ndarray
    class_names: np.ndarray

class IrisDataset:
    def load(self, test_size=None):
        iris = load_iris()


        (train_features, testing_features,train_targets, testing_targets) = skms.train_test_split(iris.data,  
                                                                                         iris.target, 
                                                                                         test_size=test_size)
        return IrisData(
            train_features=train_features,
            train_targets=train_targets,
            testing_features=testing_features,
            testing_targets=testing_targets,
            class_names=iris.target_names
        )
        