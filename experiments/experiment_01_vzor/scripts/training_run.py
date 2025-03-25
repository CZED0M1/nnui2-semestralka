import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from perceptron import Perceptron


def load_dataset():
    # Načtení datasetu breast cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # Transformace targetu: původně 0 a 1, mapujeme na -1 a 1
    y = 2 * y - 1  # 0 -> -1, 1 -> 1
    return X, y


def main():
    # Parametry trénování
    learning_rate = 0.05
    n_epochs = 50
    n_runs = 20

    X, y = load_dataset()
    # Rozdělení dat na trénovací a testovací množinu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if not os.path.exists("..\models"):
        os.mkdir("..\models")

    all_errors = []
    test_accuracies = []

    for run in range(1, n_runs + 1):
        print(f"Trénink č. {run}")
        model = Perceptron(input_dim=X_train.shape[1],
                           learning_rate=learning_rate, n_epochs=n_epochs)
        errors = model.fit(X_train, y_train)
        all_errors.append(errors)

        # Uložení modelu do složky models
        model_filename = os.path.join("..\models", f"model{run}.pkl")
        model.save_model(model_filename)

        # Testování modelu na testovací množině
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        test_accuracies.append(accuracy)

        # Uložení grafu průběhu tréninkové chyby
        plt.figure()
        plt.plot(range(1, n_epochs + 1), errors, marker='o')
        plt.title(f"Tréninková chyba – Model {run}")
        plt.xlabel("Epochy")
        plt.ylabel("Počet chyb")
        error_plot_path = os.path.join("..\images", f"model{run}_training_error.png")
        plt.savefig(error_plot_path)
        plt.close()

    # Vykreslení boxplotu testovací přesnosti všech modelů
    plt.figure()
    plt.boxplot(test_accuracies)
    plt.title(f"Testovací přesnost pro {n_runs} modelů")
    plt.ylabel("Přesnost")
    boxplot_path = os.path.join("..\images", "test_accuracy_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()

    # Výběr nejlépe performujícího modelu
    best_run = np.argmax(test_accuracies)
    print(f"Nejlepší model: model {best_run + 1} s přesností {test_accuracies[best_run]}")

    # Generování dokumentace experimentu do Markdown souboru
    if not os.path.exists("..\docs"):
        os.mkdir("..\docs")
    with open(os.path.join("..\docs", "experiment_01.md"), "w", encoding="utf-8") as f:
        f.write("# Experimenty s perceptronem\n\n")
        f.write("## Popis úlohy\n")
        f.write(
            "Cílem experimentu je trénovat jednoduchý perceptron s aktivační funkcí, která vrací **-1** pokud je potenciál menší než 0, a **1** jinak, "
            "na datasetu **Breast Cancer** načteného pomocí scikit-learn. Provedeno bylo 10 tréninkových běhů se stejnými parametry. Byl zaznamenán průběh tréninkové chyby, "
            "uloženy hodnoty vah a biasu a následně vyhodnocena testovací přesnost.\n\n")
        f.write("## Parametry trénování\n")
        f.write(f"- Počet epoch: {n_epochs}\n")
        f.write(f"- Koeficient rychlosti učení: {learning_rate}\n")
        f.write(f"- Počet běhů: {n_runs}\n\n")
        f.write("## Výsledky\n")
        f.write("### Testovací přesnosti jednotlivých modelů:\n")
        for i, acc in enumerate(test_accuracies, 1):
            f.write(f"- Model {i}: {acc}\n")
        f.write("\n")
        f.write(f"### Nejlepší model: Model {best_run + 1}\n")
        f.write("### Vizualizace\n")
        f.write("![Tréninková chyba](../images/model{}_training_error.png)\n\n".format(best_run + 1))
        f.write("![Boxplot testovací přesnosti](../images/test_accuracy_boxplot.png)\n")


if __name__ == "__main__":
    main()
