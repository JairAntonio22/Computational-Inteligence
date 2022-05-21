import numpy as np

from Model import Model
from Optimizer import Optimizer

class BlindOptimizer(Optimizer):
    n_gens: int
    n_parents: int
    n_children: int
    child_size: int
    mut_factor: int
    domain: (float, float)
    overlap: bool
    parents: np.array
    best_parents: np.array

    def __init__(self, n_gens, n_parents, n_children, mut_factor, domain, overlap):
        self.n_gens = n_gens
        self.n_parents = n_parents
        self.n_children = n_children
        self.child_size = len(domain)
        self.mut_factor = mut_factor
        self.domain = domain
        self.overlap = overlap
        self.parents = np.random.uniform(
            domain[0], domain[1], (self.n_parents, self.child_size)
        )
        self.best_parents = []

    def mutate(self, parent):
        dims = (self.n_children, self.child_size)
        children = np.full(dims, parent)
        children += np.random.uniform(-self.mut_factor, self.mut_factor, dims)
        return children

    def evaluate(self, children, model, X, y):
        scores = []

        for child in children:
            model.weights = child
            scores.append(model.score(X, y))

        return np.array(scores)

    def run(self, model, X, y):
        for _ in range(self.n_gens):
            parent = self.parents[np.random.randint(self.n_parents)]
            children = self.mutate(parent)

            if self.overlap:
                children = np.append(children, self.parents, axis=0)

            scores = self.evaluate(children, model, X, y)
            children = list(zip(scores, children))
            children.sort(reverse=True)

            self.parents = []

            for i, (_, child) in enumerate(children):
                if i < self.n_parents:
                    self.parents.append(child)
                else:
                    break

            self.parents = np.array(self.parents)
            self.best_parents.append(parent)

        model.weights = self.parents[0]
        return model
