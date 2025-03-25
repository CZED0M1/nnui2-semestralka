import numpy as np
import pickle

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # Náhodná inicializace vah a biasu
        self.weights = 0.8*np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.error_history = []  # Uchovává počet chyb v každé epoše

    def activation(self, x):
        # Aktivační funkce: vrací -1 pokud je potenciál < 0, jinak 1
        return np.where(x < 0, -1, 1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def fit(self, X, y):
        self.error_history = []
        for epoch in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update
                if update != 0.0:
                    errors += 1
            self.error_history.append(errors)
        return self.error_history

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'learning_rate': self.learning_rate,
                'n_epochs': self.n_epochs,
                'error_history': self.error_history
            }, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
            self.learning_rate = data['learning_rate']
            self.n_epochs = data['n_epochs']
            self.error_history = data['error_history']
