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

# Source - https://stackoverflow.com/a/1319675
# Posted by gahooa, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-02, License - CC BY-SA 4.0


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
        distances = []
        
        for (i, f) in enumerate(self.features):
            # distance = ((data[0] - f[0])**2 + (data[1]-f[1])**2) ** (1/2)
            distances.append((np.sqrt(np.sum((data - f)**2)), i))
            #print(f"Calculating distance from {data} to {f}... distance is {distances[i]}")

        distances.sort()
        
        #print(f" ordered distances {distances}")
        votes = []
        for v in range(self.k):
            votes.append(self.targets[distances[v][1]])
        
        winner = Counter(votes).most_common(1)[0][0]    
        #print(f"k is {self.k} votes {votes}, winner is {winner}")
        return  winner


def test_knn_with_numpy_irregular_data():
    X_train = [[1,1],[1,2],[2,1],[2,2],[5,5],[8,8],[9,8],[8,9]]
    y_train = [  0,    0,    0,    1,    0,    1,    1,    1  ]
    X_test  = [[3,3]]

    for k in range(1,6,2):
        knn = KNN_numpy(k)
        knn.fit(X_train, y_train)
        r=knn.predict(X_test)
        print(f"k={k}  result is {r}")

test_knn_with_numpy_irregular_data()