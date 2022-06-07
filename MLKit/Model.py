import numpy as np


class Model:
    weights: np.array

    def predict(self, X):
        pass

    def score(self, X, y):
        y_mean = np.mean(y)
        y_pred = self.predict(X)

        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - y_mean) ** 2)

        if v == 0:
            return 1 - u
        else:
            return 1 - (u / v)


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


class LogisticRegression(Model):
    def __init__(self, n_dims, can_scale=False, can_translate=False):
        self.can_scale = can_scale
        self.can_translate = can_translate

        if self.can_scale:
            n_dims += 1

        if self.can_translate:
            n_dims += 1

        self.weights = np.ones(n_dims + 1)

    def predict(self, X):
        start = 0

        if self.can_scale:
            scale = self.weights[start]
            start += 1

        if self.can_translate:
            translate = self.weights[start]
            start += 1

        intercept = self.weights[start + 1]
        coeffs = self.weights[start + 1:]
        results = []

        for row in X:
            result = np.dot(row, coeffs) + intercept
            result = 1 / (1 + np.exp(result))

            if self.can_scale:
                result *= scale

            if self.can_translate:
                result += translate

            results.append(result)

        results = np.array(results)

        return results
