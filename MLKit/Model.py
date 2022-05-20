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
