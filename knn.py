import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k = 3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indcies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indcies]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
'''
1. Load the data
2. Initialise the value of k
3. For getting the predicted class, iterate from 1 to total number of training data points
  3.1. Calculate the distance between test data and each row of training dataset. Here we will use Euclidean distance as our distance metric since it's the most popular method. The other distance function or metrics that can be used are Manhattan distance, Minkowski distance, Chebyshev, cosine, etc. If there are categorical variables, hamming distance can be used.
  3.2. Sort the calculated distances in ascending order based on distance values
  3.3. Get top k rows from the sorted array
  3.4. Get the most frequent class of these rows
  3.5. Return the predicted class
'''