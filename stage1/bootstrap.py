
from stage1.adapters.data import iris_loader
from stage1.knn.hybrid_knn import HybridKNNClassifier
from stage1.naive_bayes.naive_bayes import NaiveBayesClassifier
from stage1.knn.knn_with_numpy import KNNClassifier
from stage1.domain.classifier import Classifier
import numpy as np

def create_classifier(name: str) -> tuple[Classifier, np.ndarray]:
    data = iris_loader.IrisDataset().load()
    classifier = None
    if "hybrid" == name:
        classifier = HybridKNNClassifier(k=3)
    elif "knn" == name:
        classifier = KNNClassifier(k=3)
    elif "bayes" == name:
        classifier = NaiveBayesClassifier()
    else:
        raise ValueError(f"Unknown classifier: {name}")
    
    classifier.fit(data.train_features, data.train_targets)
    return (classifier, data.class_names)




