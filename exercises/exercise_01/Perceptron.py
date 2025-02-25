import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        errors = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def getErrors(self):
        return self.errors

    def saveWeights(self,filename):
        with open(filename, 'wb') as f:
            for weight in self.weights:
                f.write(str(weight) + '\n')

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_data, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def test(self, testing_data, labels):
        correct_predictions = 0
        for inputs, label in zip(testing_data, labels):
            prediction = self.predict(inputs)
            if prediction == label:
                correct_predictions += 1
        return correct_predictions / len(labels)
