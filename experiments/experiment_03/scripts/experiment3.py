import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import random

# Nastavení pro opakovatelnost
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- 1. Načtení dat ---
data = pd.read_csv('../data/experiment_data_rand.csv')
u = data.iloc[:, 2].values
y = data.iloc[:, 3].values

# --- 2. Vytvoření vstupního a výstupního datasetu ---
X, Y = [], []
for k in range(2, len(u) - 1):
    X.append([u[k], u[k - 1], y[k - 1], y[k - 2]])
    Y.append(y[k])
X = np.array(X)
Y = np.array(Y)

# --- 3. Normalizace ---
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

# --- 4. Rozdělení na trénovací a testovací sadu ---
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, shuffle=False)

# --- 5. Definice topologií ---
topologies = {
    "top1": [4, 8, 1],
    "top2": [4, 16, 1],
    "top3": [4, 32, 1],
    "top4": [4, 16, 8, 1],
    "top5": [4, 32, 16, 8, 1],
}

loss_results = {top: [] for top in topologies}
histories = {top: [] for top in topologies}
models = {}

# --- 6. Trénování každé topologie 10× ---
for top_name, layers in topologies.items():
    print(f"Trénuji {top_name}...")
    for run in range(10):
        model = Sequential()
        for i in range(1, len(layers) - 1):
            model.add(Dense(layers[i], activation='relu', input_shape=(layers[i - 1],) if i == 1 else None))
        model.add(Dense(1))  # výstupní vrstva

        model.compile(optimizer=Adam(), loss='mse')
        history = model.fit(X_train, Y_train, epochs=100, verbose=0, validation_split=0.2)

        final_loss = history.history['val_loss'][-1]
        loss_results[top_name].append(final_loss)
        histories[top_name].append(history)

        if run == 0:
            models[top_name] = model  # Ulož první běh pro případné zobrazení

# --- 7. Vykreslení boxplotu ---
plt.figure(figsize=(10, 6))
plt.boxplot([loss_results[top] for top in topologies], labels=topologies.keys())
plt.title("Boxplot validační chyby (MSE) pro různé topologie")
plt.ylabel("Validační MSE")
plt.grid(True)
plt.show()

# --- 8. Výběr nejlepší topologie a modelu ---
mean_losses = {top: np.mean(losses) for top, losses in loss_results.items()}
best_topology = min(mean_losses, key=mean_losses.get)
print(f"Nejlepší topologie: {best_topology} (průměrná val_loss = {mean_losses[best_topology]:.5f})")

best_model = models[best_topology]
Y_pred_scaled = best_model.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_denorm = scaler_Y.inverse_transform(Y_test)

# --- 9. Vyhodnocení nejlepšího modelu ---
mse = MeanSquaredError()
mse_value = mse(Y_test_denorm, Y_pred).numpy()
print(f"MSE nejlepšího modelu na testovacích datech: {mse_value:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(Y_test_denorm, label='Skutečné y')
plt.plot(Y_pred, label='Predikované y')
plt.title(f"Predikce nejlepšího modelu ({best_topology})")
plt.xlabel("Vzorek")
plt.ylabel("Výstup")
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Zápis do deníku ---
with open("denik.txt", "w") as f:
    for top, losses in loss_results.items():
        f.write(f"{top}: průměrná val_loss = {np.mean(losses):.6f}\n")
    f.write(f"\nNejlepší topologie: {best_topology}, MSE na testu: {mse_value:.6f}\n")
