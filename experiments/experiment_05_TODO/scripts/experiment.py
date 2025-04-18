import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork  # Importování třídy NeuralNetwork


def generate_data():
    # Generování dat – například funkce sinus
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = np.sin(X)
    return X, y


def plot_box_chart(history_list, labels):
    # Funkce pro vykreslení box grafu
    plt.figure(figsize=(10, 6))
    plt.boxplot(history_list, vert=True, patch_artist=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title('Porovnání trénovací chyby pro různé počty neuronů')
    plt.ylabel('Trénovací chyba')
    plt.savefig("../images/plot.png")


def run_experiment():
    X, y = generate_data()  # Generování dat
    epochs = 1000  # Počet epoch
    learning_rate = 0.001 # Koeficient učení
    hidden_neurons_list = [2, 4, 8, 16, 32]  # Počet neuronů ve skryté vrstvě pro experiment
    history_list = []  # Pro uchování průběhu trénovací chyby
    best_model = None  # Nejlepší model podle trénovací chyby

    for hidden_neurons in hidden_neurons_list:
        # Inicializace a trénování neuronové sítě
        nn = NeuralNetwork(input_size=1, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)

        print(f"\nTrénování s {hidden_neurons} neurony ve skryté vrstvě...")
        history = nn.train(X, y, epochs=epochs, verbose=False)

        # Uložení průběhu trénovací chyby
        history_list.append(history)

        # Uložení modelu
        model_filename = f"model_{hidden_neurons}_neurons.npz"
        nn.save_model(model_filename)

        # Vyhodnocení nejlepšího modelu (na základě nejnižší trénovací chyby)
        if best_model is None or np.min(history) < np.min(best_model[1]):
            best_model = (model_filename, history)

    # Vykreslení box grafu pro porovnání výkonu
    plot_box_chart(history_list, [f"{neurons} neurons" for neurons in hidden_neurons_list])

    # Načtení nejlepšího modelu
    print(f"\nNejlepší model je: {best_model[0]}")
    nn.load_model(best_model[0])

    # Předpověď na trénovacích datech
    y_pred = nn.predict(X)

    # Vykreslení výsledků
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="Skutečná funkce (sinus)")
    plt.plot(X, y_pred, label="Předpovědi nejlepšího modelu", linestyle="--")
    plt.title("Porovnání skutečné funkce a předpovědi nejlepšího modelu")
    plt.legend()
    plt.savefig("../images/plot2.png")


# Spuštění experimentu
run_experiment()
