import inspect
import strategies
from utils import load_best_genom, play


# -------------- Funkce Reaktivního Agenta ------------- #
best = load_best_genom()["genom"]

if not best:
    print("Agent is not trained yet. Please train it first.")
    exit(1)

def react_agent(my_history, other_history):
    if len(my_history) == 0:
        return 0
    idx = 2 * my_history[-1] + other_history[-1]
    return best[idx]

# -------------- Trunaj ------------- #


# seznam funkci o testování
ucastníci = [
    f for name, f in vars(strategies).items()
    if inspect.isfunction(f)
    and f.__module__ == 'strategies'
]
random_ucastnici = [always_cooperate, random_answer, tick_for_tack, react_agent, grim_trigger, pavlov, two_defects]
# funkce se mohou v seznamu i opakovat
#ucastnici = [always_cooperate, always_cooperate, random_answer, random_answer, random_answer]

STEPSNUM = 200

l = len(ucastnici)
skores = [0 for i in range(l)]

print("=========================================")
print("Turnaj")
print("hra délky:", STEPSNUM)
print("-----------------------------------------")

for i in range(l):
    for j in range(i+1, l):
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

