import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def main():
    # Cesty
    DATA_DIR = '../GTSRB'
    MODELS_DIR = '../models'
    IMAGES_DIR = '../images'

    # Vytvoření složek pokud neexistují
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Parametry
    batch_size = 128
    epochs = 1
    learning_rate = 0.001
    num_classes = 43
    num_workers = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformace
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset
    train_data = torchvision.datasets.GTSRB(
        root=DATA_DIR, split='train', download=True, transform=transform)

    test_data = torchvision.datasets.GTSRB(
        root=DATA_DIR, split='test', download=True, transform=transform)

    # Rozdělení train na train/val
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    # DataLoadery
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Definice modelu
    def build_model(topology):
        layers = []
        input_dim = 32 * 32 * 3
        for units in topology:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        layers.append(nn.Linear(input_dim, num_classes))
        return nn.Sequential(*layers)

    # Topologie
    architectures = [
        [256],
        [512, 256],
        [1024, 512, 256],
        [1024, 512, 256, 128],
        [2048, 1024, 512, 256, 128]
    ]

    # Trénink
    val_accuracies = {}
    val_losses = {}

    for idx, topology in enumerate(architectures):
        print(f"\n[INFO] Trénink topologie {idx + 1}: {topology}")
        acc_list = []
        loss_list = []

        start = time.perf_counter()  # Začátek měření času

        for run in range(10):
            model = build_model(topology).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Tréninková smyčka
            for epoch in range(epochs):
                epoch_start = time.perf_counter()  # Začátek měření času
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(inputs.size(0), -1)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                epoch_end = time.perf_counter()  # Začátek měření času
                print(f"[INFO] Trénink epochy {epoch + 1} trval {epoch_end - epoch_start:.2f} sekund.")

            # Validace
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.view(inputs.size(0), -1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = correct / total
            avg_loss = val_loss / len(val_loader)
            acc_list.append(acc)
            loss_list.append(avg_loss)

        end = time.perf_counter()  # Konec měření času
        print(f"[INFO] Trénink topologie {idx + 1} trval {end - start:.2f} sekund.")

        val_accuracies[f'Topology_{idx + 1}'] = acc_list
        val_losses[f'Topology_{idx + 1}'] = loss_list

    # Boxploty
    plt.figure()
    plt.boxplot(val_accuracies.values(), labels=val_accuracies.keys())
    plt.title('Boxplot validační přesnosti')
    plt.ylabel('Validační přesnost')
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_accuracy.png'))
    plt.close()

    plt.figure()
    plt.boxplot(val_losses.values(), labels=val_losses.keys())
    plt.title('Boxplot validační chyby (loss)')
    plt.ylabel('Validační loss')
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_loss.png'))
    plt.close()

    # Výběr nejlepší topologie
    best_topology_name = max(val_accuracies, key=lambda k: np.median(val_accuracies[k]))
    best_index = int(best_topology_name.split('_')[-1]) - 1
    best_topology = architectures[best_index]

    # Finální trénink nejlepšího modelu
    print(f"\n[INFO] Nejlepší topologie: {best_topology}")
    model = build_model(best_topology).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Uložení nejlepšího modelu
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))

    # Vyhodnocení na test datech
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Skutečné třídy')
    plt.xlabel('Predikované třídy')
    plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix.png'))
    plt.close()

    print("\n[INFO] Skript úspěšně dokončen.")

if __name__ == '__main__':
    main()