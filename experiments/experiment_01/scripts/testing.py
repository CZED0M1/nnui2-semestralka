import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from perceptron import Perceptron


def load_dataset():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Transformace: 0 -> -1, 1 -> 1
    y = 2 * y - 1
    return X, y


def test_model(model_path):
    X, y = load_dataset()
    # Rozdělení dat na trénovací a testovací množinu
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Perceptron(input_dim=X_test.shape[1])
    model.load_model(model_path)

    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Přesnost modelu {model_path} na datasetu Breast Cancer: {accuracy}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Použití: python testing.py <cesta_k_modelu>")
        exit(1)
    model_path = sys.argv[1]
    test_model(model_path)
