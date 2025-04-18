import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Inicializace velikosti vrstev a koeficientu učení
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializace váh a prahů s náhodnými hodnotami
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def activation(self, x):
        # Aplikuje aktivační funkci tanh
        return np.tanh(x)

    def activation_derivative(self, x):
        # Derivace aktivační funkce tanh (1 - tanh^2(x))
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        # Provede průchod sítí vpřed
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.y_pred = self.final_input  # Lineární výstup (bez aktivační funkce na výstupu)
        return self.y_pred

    def compute_loss(self, y_pred, y_true):
        # Vypočítá střední kvadratickou chybu (MSE)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, y_pred):
        # Zpětná propagace: výpočet gradientů a aktualizace vah a prahů

        # Chyba na výstupu
        output_error = y_pred - y_true

        # Gradienty pro váhy a biasy na výstupu
        d_output = output_error

        # Gradienty pro váhy mezi skrytou a výstupní vrstvou
        d_weights_hidden_output = np.dot(self.hidden_output.T, d_output)
        d_bias_output = np.sum(d_output, axis=0, keepdims=True)

        # Chyba ve skryté vrstvě
        hidden_error = np.dot(d_output, self.weights_hidden_output.T) * self.activation_derivative(self.hidden_input)

        # Gradienty pro váhy mezi vstupní a skrytou vrstvou
        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Aktualizace váh a prahů pomocí gradient descentu
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_hidden -= self.learning_rate * d_bias_hidden
        self.bias_output -= self.learning_rate * d_bias_output

    def train(self, X, y, epochs, verbose=False):
        # Trénování sítě přes zadaný počet epoch
        history = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Vypočítání ztráty
            loss = self.compute_loss(y_pred, y)
            history.append(loss)

            # Zpětná propagace a aktualizace vah
            self.backward(X, y, y_pred)

            # Volitelné vypisování průběžných informací
            if verbose and epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return history

    def predict(self, X):
        # Předpověď pro daný vstup
        return self.forward(X)

    def save_model(self, filepath):
        # Uložení váh a prahů do souboru (.npz)
        np.savez(filepath,
                 weights_input_hidden=self.weights_input_hidden,
                 weights_hidden_output=self.weights_hidden_output,
                 bias_hidden=self.bias_hidden,
                 bias_output=self.bias_output)

    def load_model(self, filepath):
        # Načtení váh a prahů ze souboru (.npz)
        model = np.load(filepath)
        self.weights_input_hidden = model['weights_input_hidden']
        self.weights_hidden_output = model['weights_hidden_output']
        self.bias_hidden = model['bias_hidden']
        self.bias_output = model['bias_output']

#TEST
# Definování jednoduché funkce pro generování dat (např. XOR problém)
def generate_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Vstupy
    y = np.array([[0], [1], [1], [0]])  # Výstupy (XOR)
    return X, y

# Inicializace neuronové sítě
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

# Generování trénovacích dat
X, y = generate_data()

# Trénování sítě
history = nn.train(X, y, epochs=1000, verbose=True)

# Předpověď pro trénovací data
predictions = nn.predict(X)
print("\nPředpovědi po trénování:")
print(predictions)

# Uložení modelu
nn.save_model("neural_network_model.npz")

# Načtení modelu
nn.load_model("neural_network_model.npz")

# Předpovědi po načtení modelu
predictions_after_load = nn.predict(X)
print("\nPředpovědi po načtení modelu:")
print(predictions_after_load)
