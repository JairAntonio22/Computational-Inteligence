import random
import sys
sys.path.append('MLKit')

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import MLKit

if __name__ == '__main__':
    st.sidebar.title('Dataset generation')

    dataset_size = st.sidebar.slider('dataset size', 1, 100, 30)
    noise_factor = st.sidebar.slider('noise', 0.0, 10.0, 1.0)
    intercept = st.sidebar.slider('intercept', -5.0, 5.0, -2.0)
    slope = st.sidebar.slider('slope', -5.0, 5.0, 1.0)
    x_domain = 10

    data = np.random.uniform(-x_domain, x_domain, (dataset_size, 1))
    labels = [
        intercept + slope * value[0] + random.uniform(-noise_factor, noise_factor)
        for value in data
    ]

    st.sidebar.title('Optimizer parameters')

    optimizer = MLKit.BlindOptimizer(
        n_gens = st.sidebar.slider('number of generations', 1, 500, 100),
        n_parents = st.sidebar.slider('number of parents', 1, 10, 5),
        n_children = st.sidebar.slider('number of children', 1, 20, 10),
        mut_factor = st.sidebar.slider('mutation factor', 0.01, 2.5, 0.5),
        domain = (-10, 10),
        overlap = st.sidebar.checkbox('generational overlap', True)
    )

    regressor = optimizer.run(MLKit.LinearRegression(1), data, labels)
    score = regressor.score(data, labels)

    st.title('Linear Regressor')
    st.write('Model to train')
    st.latex('h(x_i, \\beta) = \\beta_0 + \\sum_{i=1}^n \\beta_i \cdot x_i ')

    st.write('Original function')
    st.latex(f'f(x) = {intercept: 3.2f} + {slope: 3.2f} \cdot x')

    r_intercept = regressor.weights[0]
    r_slope = regressor.weights[1]

    st.write('Solution found')
    st.latex(f'f(x) = {r_intercept: 3.2f} + {r_slope: 3.2f} \cdot x')

    st.write('Score')
    st.latex(f'R^2 \\approx {score: 8.4}')

    col1, col2 = st.columns(2)

    fig, ax = plt.subplots()

    X = np.linspace([-x_domain], [x_domain])
    Y = regressor.predict(X)

    plt.title('Regression')
    plt.ylabel('x')
    plt.xlabel('y')

    ax.scatter(data, labels)
    ax.plot(X, Y, c='r')

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
