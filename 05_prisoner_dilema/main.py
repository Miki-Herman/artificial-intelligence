import inspect
import strategies
from utils import load_best_genom, play


# -------------- Funkce Reaktivního Agenta ------------- #
best = load_best_genom()["genom"]

if not best:
    print("Agent is not trained yet. Please train it first.")
    exit(1)

def zrada(my_history, other_history):
    round_num = len(my_history)

    # 1. Úvodní fáze: poznávání soupeře
    if round_num < 3:
        return 0  # spolupracuj pro navázání důvěry
    elif round_num < 10:
        if other_history[-1] == 1:
            return 1  # test: pokud mě zradí brzo, odpověz zradou
        return 0  # jinak dál spolupracuj

    # 2. Detekce defektora
    recent_defections = other_history[-5:].count(1)
    total_defections = other_history.count(1)
    if total_defections / round_num > 0.3 and recent_defections >= 3:
        return 1  # zraď, pokud je soupeř defektor

    # 3. Odpouštění
    if other_history[-2:] == [0, 0]:
        return 0  # odpustíme po 2 spolupracích

    # 4. Genomová tabulka (paměť 2 tahy zpět)
    idx = 8 * my_history[-2] + 4 * my_history[-1] + 2 * other_history[-2] + other_history[-1]
    return best[idx]

# -------------- Trunaj ------------- #


# seznam funkci o testování
ucastnici = [
    f for name, f in vars(strategies).items()
    if inspect.isfunction(f)
    and f.__module__ == 'strategies'
]

ucastnici.append(zrada)

STEPSNUM = 200

l = len(ucastnici)
skores = [0 for i in range(l)]

print("=========================================")
print("Turnaj")
print("hra délky:", STEPSNUM)
print("-----------------------------------------")

for i in range(l):
    for j in range(i, l):
        f1 = ucastnici[i]
        f2 = ucastnici[j]
        skore1, skore2 = play(f1, f2, STEPSNUM)
        print(f1.__name__, "x", f2.__name__, " ", skore1, ":", skore2)
        skores[i] += skore1
        skores[j] += skore2

print("=========================================")
print("= Výsledné pořadí")
print("-----------------------------------------")

# setrideni indexu vysledku
index = sorted(range(l), key=lambda k: skores[k])

poradi = 1
for i in index:
    f = ucastnici[i]
    print(poradi, ".", f.__name__, ":", skores[i])
    poradi += 1

