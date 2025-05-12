---
title: "Experiment 05"
author: "Dominik Lopauer"
date: "2025-05-09"
---

# Experiment 03: Srovnání různých topologií CNN na úloze klasifikace dopravních značek (GTSRB)

## Popis úlohy
Cílem experimentu bylo porovnat různé topologie dopředných neuronových sítí (FFNN) při klasifikaci dopravních značek z datasetu GTSRB. Síť byla trénována na 43 třídách dopravních značek. Experiment hodnotí výkonnost modelů pomocí metriky **validační přesnosti** a **validační ztráty**.

Každá topologie byla natrénována **10×** a výsledky byly porovnány pomocí boxplotů.

## Parametry trénování
- **Počet epoch:** 1O
- **Koeficient učení:** 0.001
- **Velikost dávky:** 128
- **Optimalizátor:** Adam
- **Ztrátová funkce:** CrossEntropyLoss
- **Velikost vstupu:** 32×32×3 (obrázky)
- **Počet výstupních tříd:** 43
- **Transformace:** Resize, ToTensor, Normalize

## Porovnávané topologie
- **Topologie 1:** [32,3]
- **Topologie 2:** [(64, 3), (32, 3)]
- **Topologie 3:** [(128, 3), (64, 3)]
- **Topologie 4:** [(64, 3), (128, 3), (64, 3)]
- **Topologie 5:** [(128, 3), (128, 3), (64, 3)]

## Výsledky – boxploty

![Boxplot validační přesnosti](../images/boxplot_cnn_accuracy.png)

![Boxplot validační chyby](../images/boxplot_cnn_loss.png)

Boxploty ukazují rozložení přesnosti a chyb pro každou topologii na validační množině. Nejvyšší medián přesnosti byla dosažena u **Topologie 3**, což ji činí nejlepší volbou pro tuto klasifikační úlohu.

## Popis nejlepšího modelu
Nejlepšího výsledku bylo dosaženo s topologií **[(128, 3), (64, 3)]**. Tato architektura obsahuje pět skrytých vrstev s postupně se zmenšujícím počtem neuronů. Dosáhla nejvyšší mediánové přesnosti při validaci a ukázala schopnost dobře generalizovat.

Model byl po finálním natrénování uložen jako `best_model.pt`.

![confusion_matrix](../images/confusion_matrix_cnn.png)

## Závěr

- Nejlepší výsledek byl dosažen u topologie 4 (nejhlubší síť).
- Přesnost na testovacích datech dosáhla okolo **98 %**.
---
