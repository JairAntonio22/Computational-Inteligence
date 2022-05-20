import numpy as np

from Model import Model

class LinearRegression(Model):
    def __init__ (self, n_dims):
        self.weights = np.ones(n_dims + 1)

    def predict(self, X):
        intercept = self.weights[0]
        coeffs = self.weights[1:]
        results = []

        for row in X:
            result = np.dot(row, coeffs) + intercept
            results.append(result)

        results = np.array(results)

        return results
