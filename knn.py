import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

class MyKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Algorithm implementation using NumPy.
    """
    def __init__(self, n_neighbors=3, metric="euclidean", weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores the training data.
        """
        # Store X and y in self.X_train and self.y_train.
        self.X_train = X
        self.y_train = y
        # Return self for sklearn compatibility
        return self

    def _calculate_distances(self, x):
        """ 
        Calculates the distance from sample x to all training data.
        """
        if self.metric == "euclidean":
            # Euclidean distance: sqrt(sum((x_i - y_i)^2))
            return np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        elif self.metric == "manhattan":
            # Manhattan distance: sum(|x_i - y_i|)
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _predict_one(self, x):
        """
        Predicts the label for a single sample x.
        """
        # 1. Calculate distances
        distances = self._calculate_distances(x)

        # 2. Find the k nearest neighbors (indices and distances)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        # 3. Perform voting
        if not self.weighted:
            # Unweighted mode: Simple voting (finding the most frequent)
            # Use Counter to find the simplest most frequent item
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else:
            # Weighted mode: Weighted voting (based on 1 / distance)
            # Add a very small value (epsilon) to prevent division by zero
            epsilon = 1e-10
            weights = 1 / (k_nearest_distances + epsilon)

            # Sum of weights for each class
            class_scores = {}
            for label, weight in zip(k_nearest_labels, weights):
                if label not in class_scores:
                    class_scores[label] = 0
                class_scores[label] += weight

            # Return the class with the highest score (sum of weights)
            return max(class_scores, key=class_scores.get)

    def predict(self, X):
        """
        Predicts labels for a set of samples X.
        """
        # TODO:
        # 1. Call the _predict_one method for each sample in X.
        # 2. Return the results as a NumPy array.
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model has not been fit() yet.")

        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)
