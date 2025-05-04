import numpy as np
import random
import time
import os
from deap import base, creator, tools, algorithms

# Bludiště
MAZE = [
    "#########",
    "#     #G#",
    "# ### # #",
    "# #     #",
    "# ##### #",
    "#       #",
    "#########"
]
maze = [list(row) for row in MAZE]
start_pos = (1, 1)
goal_pos = (7, 1)

# ---------- Agent a senzorické vstupy ----------
def get_sensor_inputs(position):
    x, y = position
    walls = [
        int(maze[y-1][x] != '#'),  # nahoru
        int(maze[y+1][x] != '#'),  # dolů
        int(maze[y][x-1] != '#'),  # vlevo
        int(maze[y][x+1] != '#')   # vpravo
    ]
    dx = (goal_pos[0] - x) / len(maze[0])
    dy = (goal_pos[1] - y) / len(maze)
    return walls + [dx, dy]

# ---------- Neuronová síť ----------
def nn_function(inp, genome):
    inp = np.array(inp)
    weights = np.array(genome).reshape((6, 4))  # 6 vstupů × 4 výstupy
    out = np.dot(inp, weights)
    return np.tanh(out)

# ---------- Pohyb agenta ----------
def nn_navigate_me(position, genome):
    inp = get_sensor_inputs(position)
    out = nn_function(inp, genome)
    move = np.argmax(out)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # ↑ ↓ ← →
    new_x = position[0] + directions[move][0]
    new_y = position[1] + directions[move][1]
    if maze[new_y][new_x] != '#':
        return (new_x, new_y)
    return position  # zůstaň stát, pokud narážíš

# ---------- Vizualizace ----------
def print_maze(agent_pos, goal_pos):
    os.system('cls' if os.name == 'nt' else 'clear')
    for y, row in enumerate(maze):
        row_str = ''
        for x, cell in enumerate(row):
            if (x, y) == agent_pos:
                row_str += 'A'
            elif (x, y) == goal_pos:
                row_str += 'G'
            else:
                row_str += cell
        print(row_str)
    time.sleep(0.25)

# ---------- Simulace jednoho běhu agenta ----------
def simulate_agent(genome, max_steps=50, visualize=False):
    pos = start_pos
    for _ in range(max_steps):
        if visualize:
            print_maze(pos, goal_pos)
        pos = nn_navigate_me(pos, genome)
        if pos == goal_pos:
            return True
    return False

# ---------- Fitness funkce ----------
def evaluate(genome):
    success_count = 0
    trials = 5
    for _ in range(trials):
        if simulate_agent(genome):
            success_count += 1
    return success_count / trials,  # fitness = úspěšnost (0.0 až 1.0)

# ---------- DEAP konfigurace ----------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, 24)  # 6x4
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------- Tréninkový proces ----------
def train(pop_size=100, ngen=50, target=0.9):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    for gen in range(ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=pop_size)
        hof.update(pop)
        record = stats.compile(pop)
        print(f"Generace {gen}: průměrná úspěšnost = {record['avg']:.2f}, max = {record['max']:.2f}")
        if hof[0].fitness.values[0] >= target:
            print("✅ Cíl dosažen – agent dosahuje cíle s alespoň 90% úspěšností!")
            break
    return hof[0]

# ---------- Spusť trénink ----------
best_genome = train()

# ---------- Test nejlepšího genomu ----------
print("\n🎯 Test nejlepšího genomu s vizualizací:")
successes = sum(simulate_agent(best_genome, visualize=False) for _ in range(20))
print(f"Úspěšnost: {successes}/20 ({successes * 5}%)")

# ---------- Vizualizuj jeden úspěšný běh ----------
print("\n👁 Vizualizace běhu nejlepšího agenta:")
simulate_agent(best_genome, visualize=True)