#     Build a NaiveBayesClassifier class with:
# - fit(X_train, y_train)
#     → compute prior probability for each class
#     → compute mean and std of each feature per class

# - predict(X_test)
#     → for each test point, compute the probability 
#        it belongs to each class
#     → return the class with highest probability

# Use the Gaussian probability formula for each feature:
# P(x|class) = (1 / sqrt(2π σ²)) * exp(−(x−μ)² / 2σ²)

# The final probability for a class is:
# P(class) * P(x1|class) * P(x2|class) * ...
# (this is the "naïve" part — multiply independently)

# Test on the same X_train/y_train from your k-NN.
# Expected output: [0, 1] for X_test = [[2,2],[7,7]]

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        pass
    def fit(self, features, targets):
        features = np.array(features)
        targets = np.array(targets)

        total = len(targets)
        classes = np.unique(targets)
        self.class_probability = {}

        for c in classes:
            self.class_probability[c] = {"prior":np.sum(targets == c) / total}
            class_features = features[targets == c]

            self.class_probability[c]["mean"] = np.mean(class_features, axis=0)  # mean of each column
            self.class_probability[c]["std"]  = np.std(class_features, axis=0)   # std of each column

    @staticmethod        
    def gaussian(std, mean, data):
        std = np.where(std == 0, 1e-9, std)
        return (1 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-(data - mean) ** 2 / (2 * std ** 2))

    def predict(self, data):
        result=[]
        for d in data:
            probs = {}
            for c in self.class_probability.keys():
                std = self.class_probability[c]["std"]
                mean = self.class_probability[c]["mean"]
                probs[c] = np.prod(NaiveBayesClassifier.gaussian(std,mean,d)) * self.class_probability[c]["prior"]
            result.append(max(probs, key=probs.get))
        return result
    
    
