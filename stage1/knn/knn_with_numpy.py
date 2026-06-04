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

class KNN_numpy:

    def __init__(self,  k=1):
        self.features=None
        self.targets=None
        self.k = k

    def fit(self, features, targets):
        self.features = np.array(features)
        self.targets = np.array(targets)

    def predict(self, input):
        result = []
        for i in input:
            nearest = self.predict_one(i)
            result.append(nearest)
        return result

    def predict_one(self, data):
        local_k = self.k
     
        if len(self.targets) < self.k:
            local_k = len(self.targets) 
        distances = []
        
        for (i, f) in enumerate(self.features):
            distances.append((np.sqrt(np.sum((data - f)**2)), i))

        distances.sort()
        
        votes = []
        for v in range(local_k):
            votes.append(self.targets[distances[v][1]])
        
        winner = Counter(votes).most_common(1)[0][0]
        return  winner
