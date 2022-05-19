import math
import random

import streamlit as st
import matplotlib.pyplot as plt

class Model:
    weights = []

    def predict(self, X):
        pass

    def score(self, X, y):
        y_mean = sum(y) / len(y)
        y_pred = self.predict(X)

        u = sum(map(lambda true, pred: (true - pred) ** 2, y, y_pred))
        v = sum(map(lambda true: (true - y_mean) ** 2, y))

        if v == 0:
            return 1 - u
        else:
            return 1 - (u / v)

class LinearRegressor(Model):
    def __init__ (self, n_dims):
        self.weights = [0] * (n_dims + 1)

    def predict(self, X):
        intercept = self.weights[0]
        coeffs = self.weights[1:]

        return [
            sum(map(lambda x, y: x * y, row, coeffs)) + intercept
            for row in X
        ]

class Optimizer:
    def run(self, model, X, y):
        pass

def clamp(value, lower, upper):
    return max(lower, min(value, upper))

class BlindOptimizer(Optimizer):
    n_gens: int
    n_parents: int
    n_children: int
    mut_factor: int
    domain: int
    overlap: bool
    parents: list
    best_parents: list
    best_evals: list

    def __init__(self, n_gens, n_parents, n_children, mut_factor, domain, overlap):
        self.n_gens = n_gens
        self.n_parents = n_parents
        self.n_children = n_children
        self.mut_factor = mut_factor
        self.domain = domain
        self.overlap = overlap

        self.parents = [
            [random.uniform(lower, upper) for lower, upper in self.domain]
            for _ in range(self.n_parents)
        ]

        self.best_parents = []
        self.best_evals = []

    def mutate(self, parent):
        children = [parent.copy() for _ in range(self.n_children)]

        for i, _ in enumerate(children):
            for j, _ in enumerate(children[i]):
                lower, upper = self.domain[j]
                children[i][j] += random.uniform(-self.mut_factor, self.mut_factor)
                children[i][j] = clamp(children[i][j], lower, upper)

        return children

    def evaluate(self, children, model, X, y):
        evals = []

        for child in children:
            model.weights = child
            evals.append(model.score(X, y))

        return evals

    def run(self, model, X, y):
        for _ in range(self.n_gens):
            parent = random.choice(self.parents)
            children = self.mutate(parent)

            if self.overlap:
                children += self.parents

            evals = self.evaluate(children, model, X, y)

            children = list(zip(evals, children))
            children.sort(reverse=True)

            self.parents = [child for _, child in children][:self.n_parents]
            self.best_parents.append(parent.copy())

            model.weights = parent
            self.best_evals.append(model.score(X, y))

        model.weights = self.best_parents[-1]
        return model

if __name__ == '__main__':
    random.seed(0)

    st.sidebar.title('Dataset generation')

    dataset_size = st.sidebar.slider('dataset size', 1, 100, 30)
    noise_factor = st.sidebar.slider('noise', 0.0, 10.0, 1.0)
    intercept = st.sidebar.slider('intercept', -5.0, 5.0, -2.0)
    slope = st.sidebar.slider('slope', -5.0, 5.0, 1.0)
    x_domain = 10

    data = [[random.uniform(-x_domain, x_domain)] for x in range(dataset_size)]
    labels = [
        intercept + slope * value[0] + random.uniform(-noise_factor, noise_factor)
        for value in data
    ]

    st.sidebar.title('Optimizer parameters')

    optimizer = BlindOptimizer(
        n_gens = st.sidebar.slider('number of generations', 1, 500, 100),
        n_parents = st.sidebar.slider('number of parents', 1, 10, 5),
        n_children = st.sidebar.slider('number of children', 1, 20, 10),
        mut_factor = st.sidebar.slider('mutation factor', 0.01, 2.5, 0.5),
        domain = [(-10, 10), (-10, 10)],
        overlap = st.sidebar.checkbox('generational overlap', True)
    )

    regressor = optimizer.run(LinearRegressor(1), data, labels)
    score = regressor.score(data, labels)

    st.title('Linear Regressor')
    st.write('Model to train')
    st.latex('h(x_i, \\beta) = \\beta_0 + \\sum_{i=1}^n \\beta_i \cdot x_i ')

    st.write('Original function')
    st.latex(f'f(x) = {intercept: 3.2f} + {slope: 3.2f} \cdot x')

    r_intercept = regressor.weights[0]
    r_slope = regressor.weights[1]

    st.write('Solution found')
    st.latex(f'h(x) = {r_intercept: 3.2f} + {r_slope: 3.2f} \cdot x')

    st.write('Score')
    st.latex(f'R^2 \\approx {score: 8.4}')

    col1, col2 = st.columns(2)

    fig, ax = plt.subplots()

    dx = (2 * x_domain) / 10
    x = -x_domain
    X = []

    while x <= x_domain:
        X.append([x])
        x += dx

    Y = regressor.predict(X)

    plt.title('Regression')
    plt.ylabel('x')
    plt.xlabel('y')

    ax.plot(X, Y)
    ax.scatter(data, labels)

    col1.pyplot(fig)

    fig, ax = plt.subplots()

    gens = [i for i in range(optimizer.n_gens)]
    intercepts = [w[0] for w in optimizer.best_parents]
    slopes = [w[1] for w in optimizer.best_parents]

    plt.title('Training performance')
    plt.ylabel('values')
    plt.xlabel('generations')

    ax.plot(gens, [intercept for _ in range(optimizer.n_gens)], 'r--')
    ax.plot(gens, [slope for _ in range(optimizer.n_gens)], 'k--')
    ax.plot(gens, intercepts)
    ax.plot(gens, slopes)

    plt.legend(['original intercept', 'original slope', 'intercept', 'slope'])

    col2.pyplot(fig)
