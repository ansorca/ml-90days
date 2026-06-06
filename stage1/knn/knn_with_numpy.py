# Build a KNNClassifier class with:
# - fit(X_train, y_train)  — just store the data
# - predict(X_test, k)     — for each test point, find k 
#                            nearest neighbors by Euclidean 
#                            distance, return majority class

# Test it on this data:
# X_train = [[1,2],[2,3],[3,1],[6,7],[7,8],[8,6]]
# y_train = [0, 0, 0, 1, 1, 1]
# X_test  = [[2,2],[7,7]]

# Expected output: [0, 1]
# Implemented from scratch as part of ml-90days study plan
# Author: Jose Antonio Soriano Camarillo


import numpy as np
from collections import Counter

class KNNClassifier:

    def __init__(self,  k=1):
        self.features=None
        self.targets=None
        self.k = k

    def fit(self, features, targets):
        self.features = np.array(features)
        self.targets = np.array(targets)

    def predict(self, input):
        return  np.array([self.predict_one(x) for x in input])
        

    def predict_one(self, data):
        local_k = min(len(self.targets), self.k)

        distances = np.sqrt(np.sum((data - self.features) ** 2, axis=1))

        
        # Get the indices of the 'k' smallest distances
        # np.argsort returns the indices that would sort the array
        k_indices = np.argsort(distances)[:local_k]
        k_nearest_labels = self.targets[k_indices]

        return Counter(k_nearest_labels).most_common(1)[0][0]
        
