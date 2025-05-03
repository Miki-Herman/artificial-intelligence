from deap import base, creator, tools, algorithms
from utils import play, write_best_genom
import strategies
import random
import inspect

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# načtení všech strategii
test_ucastnici = [
    f for name, f in vars(strategies).items()
    if inspect.isfunction(f)
       and f.__module__ == 'strategies'
]

# Strategie pro evoluci k získání "nejlepšího" genomu
def strategy_from_genom(genom):
    def strategy(my_history, opponent_history):
        round_num = len(my_history)

        # 1. Úvodní fáze: poznávání soupeře
        if round_num < 3:
            return 0  # spolupracuj pro navázání důvěry
        elif round_num < 10:
            if opponent_history[-1] == 1:
                return 1  # test: pokud mě zradí brzo, odpověz zradou
            return 0  # jinak dál spolupracuj

        # 2. Detekce defektora
        recent_defections = opponent_history[-5:].count(1)
        total_defections = opponent_history.count(1)
        if total_defections / round_num > 0.3 and recent_defections >= 3:
            return 1  # zraď, pokud je soupeř defektor

        # 3. Odpouštění
        if opponent_history[-3:] == [0, 0, 0]:
            return 0  # odpustíme po 3 spolupracích

        # 4. Genomová tabulka (paměť 2 tahy zpět)
        idx = 8 * my_history[-2] + 4 * my_history[-1] + 2 * opponent_history[-2] + opponent_history[-1]
        return genom[idx]
    return strategy

# fitness function, která vrátí skóre, které potomek získal
def fitness_function(individual):
    moje_strategie = strategy_from_genom(individual)
    total_score = 0
    for protivnik in test_ucastnici:
        s, _ = play(moje_strategie, protivnik, 200)
        total_score += s
    return total_score / len(test_ucastnici),

# připravit deap toolbox
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --------------- Evoluce ------------------ #
pop = toolbox.population(n=200)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)

best = tools.selBest(pop, 1)[0]
best_score = toolbox.evaluate(best)[0]

# vypsání výsledků
print("------------------------------")
print("Evoluční generování dokončeno")
print("------------------------------")

print("Data")
print(f"Genom: {best}")
print(f"Score: {best_score}")
print("------------------------------")

print("Zapisuji genom ...")
write_best_genom(best, test_ucastnici, best_score)
print("Dokončeno ...")
