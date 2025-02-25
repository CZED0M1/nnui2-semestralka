from experiments.experiment_01.data.Iris import iris_data
from experiments.experiment_01.models.Perceptron import Perceptron

data = iris_data()
perceptron = Perceptron(input_size=4)

# Train the Perceptron on the training data
//perceptron.train(data.features[0], data.features[1])
# Test the perceptron
#accuracy = perceptron.test(data.features, data.targets)

#mění se váhy sami protože random
# evidovat do deníku

print(f"Accuracy: {accuracy * 100:.2f}%")