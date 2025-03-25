---
title: "Experiment 02"
author: "Dominik Lopauer"
date: "2025-03-25"
---

# Experiment 02: Trénování neuronové sítě s různým počtem neuronů ve skryté vrstvě

## Popis úlohy
Cílem tohoto experimentu bylo porovnat výkon neuronové sítě s různým počtem neuronů ve skryté vrstvě. Úloha zahrnovala trénování modelu na datech generovaných pomocí funkce sinus a hodnocení výkonnosti modelu na základě trénovací chyby (MSE). Po každém trénování byla uložena váha modelu a testována na trénovacích datech.

## Parametry trénování
- **Počet epoch:** 1000
- **Koeficient učení:** 0.001
- **Počet neuronů ve skryté vrstvě:** 2, 4, 8, 16, 32
- **Vstupní velikost:** 1 (jednoduchý vektor)
- **Výstupní velikost:** 1 (jednoduchý vektor)

## Popis nejlepšího modelu
Nejlepší model byl dosažen s **32 neurony ve skryté vrstvě**, kde byla trénovací chyba nejnižší. Po trénování modelu byly získány následující výsledky:

### Hodnoty vah a prahů:
Model s 32 neurony ve skryté vrstvě dosáhl velmi nízké hodnoty ztráty během trénování. Trénovací chyba klesla až na hodnoty blízké nule, což je indikátor dobré konvergence modelu.

### Graf trénovací chyby
Vývoj trénovací chyby během trénování je zobrazen v grafu níže. Pozorujeme, jak se ztráta rychle snižuje na velmi nízké hodnoty.

![Trénovací chyba](plot.png)

### Předpovědi na trénovacích datech
Po dokončení trénování modelu s 32 neurony byly předpovědi na trénovacích datech velmi přesné:

Předpovědi po trénování:

[[-9.62063762e-13]
 [ 1.00000000e+00]
 [ 1.00000000e+00]
 [-9.49851309e-13]]

Předpovědi po načtení modelu:

[[-9.62063762e-13]
 [ 1.00000000e+00]
 [ 1.00000000e+00]
 [-9.49851309e-13]]
 
 ![Porovnání skutečné funkce a předpovědi nejlepšího modelu](plot2.png)


## Výsledky na testovací množině
Testování modelu na trénovacích datech ukázalo, že model s 32 neurony v skryté vrstvě dosahuje vysoké přesnosti, s velmi nízkou chybou mezi skutečnými hodnotami a předpověďmi modelu.

## Závěr
Model s 32 neurony ve skryté vrstvě se ukázal jako nejlepší, protože vykazoval nejlepší výkon na trénovacích datech. Bylo pozorováno, že jak se zvyšoval počet neuronů, model byl schopný lépe zachytit složitější vzory v datech, což vedlo k nižší chybě.

Nejlepší model je uložen jako `model_32_neurons.npz`.
