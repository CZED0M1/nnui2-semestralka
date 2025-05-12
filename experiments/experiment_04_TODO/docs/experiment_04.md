---
title: "Experiment 04"
author: "Dominik Lopauer"
date: "2025-05-09"
---

# Experiment 04: Srovnání různých topologií FFNN na úloze klasifikace dopravních značek (GTSRB)

## Popis úlohy
Cílem experimentu bylo porovnat různé topologie dopředných neuronových sítí (FFNN) při klasifikaci dopravních značek z datasetu GTSRB. Síť byla trénována na 43 třídách dopravních značek. Experiment hodnotí výkonnost modelů pomocí metriky **validační přesnosti** a **validační ztráty**.

Každá topologie byla natrénována **10×** a výsledky byly porovnány pomocí boxplotů.

## Parametry trénování
- **Počet epoch:** 1 (z důvodu časové náročnosti)
- **Koeficient učení:** 0.001
- **Velikost dávky:** 128
- **Optimalizátor:** Adam
- **Ztrátová funkce:** CrossEntropyLoss
- **Velikost vstupu:** 32×32×3 (obrázky)
- **Počet výstupních tříd:** 43
- **Transformace:** Resize, ToTensor, Normalize

## Porovnávané topologie
- **Topologie 1:** [256]
- **Topologie 2:** [512, 256]
- **Topologie 3:** [1024, 512, 256]
- **Topologie 4:** [1024, 512, 256, 128]
- **Topologie 5:** [2048, 1024, 512, 256, 128]

## Výsledky – boxploty

![Boxplot validační přesnosti](../images/boxplot_accuracy.png)

![Boxplot validační chyby](../images/boxplot_loss.png)

Boxploty ukazují rozložení přesnosti a chyb pro každou topologii na validační množině. Nejvyšší medián přesnosti byla dosažena u **Topologie 2**, což ji činí nejlepší volbou pro tuto klasifikační úlohu.

## Popis nejlepšího modelu
Nejlepšího výsledku bylo dosaženo s topologií **[512, 256]**. Tato architektura obsahuje pět skrytých vrstev s postupně se zmenšujícím počtem neuronů. Dosáhla nejvyšší mediánové přesnosti při validaci a ukázala schopnost dobře generalizovat.

Model byl po finálním natrénování uložen jako `best_model.pt`.

![confusion_matrix](../images/confusion_matrix.png)

## Závěr

- FFNN dokázaly klasifikovat dopravní značky z GTSRB s přesností okolo 73 % na testu
- Topologie s dvěma skrytými vrstvami ([512, 256]) poskytla nejlepší výsledky
---

