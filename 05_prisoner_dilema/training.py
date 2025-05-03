from deap import base, creator, tools, algorithms
from utils import play, write_best_genom, load_best_genom
import strategies
import random
import inspect

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

test_ucastnici = [
    f for name, f in vars(strategies).items()
    if inspect.isfunction(f)
       and f.__module__ == 'strategies'
]


def strategy_from_genom(genom):
    def strategy(my_history, opponent_history):
        if len(my_history) < 2:
            return 0  # první tah kooperace
        idx = 8 * my_history[-2] + 4 * my_history[-1] + 2 * opponent_history[-2] + opponent_history[-1]
        return genom[idx]
    return strategy

def fitness_function(individual):
    moje_strategie = strategy_from_genom(individual)
    total_score = 0
    for protivnik in test_ucastnici:
        s, _ = play(moje_strategie, protivnik, 20)
        total_score += s
    return min(play(moje_strategie, p, 200)[0] for p in test_ucastnici),

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --------------- Evoluce ------------------ #
pop = toolbox.population(n=100)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)

best = tools.selBest(pop, 1)[0]
best_score = toolbox.evaluate(best)[0]

write_best_genom(best, test_ucastnici, best_score)
print(load_best_genom())