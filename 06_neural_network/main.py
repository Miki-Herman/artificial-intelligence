import numpy as np
import random
import time
import os
from deap import base, creator, tools

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

# Maximální vzdálenost v bludišti pro normalizaci
max_distance = len(maze) + len(maze[0])

# ---------- senzorické vstupy ----------
def get_sensor_inputs(position):
    x, y = position
    # Vylepšené senzory - detekce vzdálenosti ke zdi v každém směru (maximálně 3 pole)
    wall_distances = [
        min([i for i in range(1, 4) if y-i < 0 or maze[y-i][x] == '#'] or [3]),  # nahoru
        min([i for i in range(1, 4) if y+i >= len(maze) or maze[y+i][x] == '#'] or [3]),  # dolů
        min([i for i in range(1, 4) if x-i < 0 or maze[y][x-i] == '#'] or [3]),  # vlevo
        min([i for i in range(1, 4) if x+i >= len(maze[0]) or maze[y][x+i] == '#'] or [3])   # vpravo
    ]
    # Normalizace vzdáleností
    wall_distances = [d/3 for d in wall_distances]

    # Euklidovská vzdálenost k cíli (normalizovaná)
    dx = goal_pos[0] - x
    dy = goal_pos[1] - y
    distance_to_goal = np.sqrt(dx**2 + dy**2) / max_distance

    # Směr k cíli (normalizovaný)
    angle_to_goal = np.arctan2(dy, dx) / np.pi  # Rozsah -1 až 1

    # Přidání informace o tom, zda je agent na křižovatce
    junction = sum(1 for d in wall_distances if d > 0) > 2

    # Předchozí pohyby (implementace paměti)
    memory = getattr(get_sensor_inputs, 'memory', [(0, 0)] * 3)
    last_moves = [(position[0] - pos[0], position[1] - pos[1]) for pos in memory]
    # Použijeme pouze poslední 3 pohyby (pouze x souřadnice) pro snížení dimenze na 10
    memory_flat = [m[0]/1 for m in last_moves]  # normalizované, pouze x souřadnice

    # Aktualizace paměti
    memory.pop(0)
    memory.append(position)
    get_sensor_inputs.memory = memory

    return wall_distances + [distance_to_goal, angle_to_goal, int(junction)] + memory_flat

# Inicializace paměti
get_sensor_inputs.memory = [start_pos] * 3

# ---------- Neuronová síť se 3 vrstvami ----------
def nn_function(inp, genome):
    """Vstupní vektor má 10 hodnot:
    - 4 vzdálenosti ke zdem
    - vzdálenost k cíli
    - úhel k cíli
    - příznak křižovatky
    - 3 předchozí pohyby (pouze x souřadnice)"""
    inp = np.array(inp)

    # Pevná struktura neuronové sítě
    input_size = 10  # Pevně stanoveno podle našich vstupů
    hidden1_size = 12
    hidden2_size = 8
    output_size = 4

    # Velikost jednotlivých částí genomu
    weights1_size = input_size * hidden1_size
    bias1_size = hidden1_size
    weights2_size = hidden1_size * hidden2_size
    bias2_size = hidden2_size
    weights3_size = hidden2_size * output_size
    bias3_size = output_size

    # Určení indexů pro rozdělení genomu
    idx1 = weights1_size
    idx2 = idx1 + bias1_size
    idx3 = idx2 + weights2_size
    idx4 = idx3 + bias2_size
    idx5 = idx4 + weights3_size

    # Extrakce vah a biasů z genomu
    w1 = np.array(genome[:idx1]).reshape((input_size, hidden1_size))
    b1 = np.array(genome[idx1:idx2])
    w2 = np.array(genome[idx2:idx3]).reshape((hidden1_size, hidden2_size))
    b2 = np.array(genome[idx3:idx4])
    w3 = np.array(genome[idx4:idx5]).reshape((hidden2_size, output_size))
    b3 = np.array(genome[idx5:])

    # Dopředný průchod sítí s aktivačními funkcemi
    # ReLU pro skryté vrstvy
    hidden1 = np.maximum(0, np.dot(inp, w1) + b1)  # ReLU
    hidden2 = np.maximum(0, np.dot(hidden1, w2) + b2)  # ReLU
    # Aplikace tanh na výstupní vrstvu pro rozsah -1 až 1
    output = np.tanh(np.dot(hidden2, w3) + b3)

    return output

# ---------- Softmax pro výstupy ----------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ---------- Pohyb agenta ----------
def nn_navigate_me(position, genome):
    inp = get_sensor_inputs(position)
    out = nn_function(inp, genome)

    # Použití softmax na výstupy pro získání rozdělení pravděpodobnosti
    probs = softmax(out)

    # Přiřadíme směry k pravděpodobnostem
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # ↑ ↓ ← →

    # Výběr směru podle pravděpodobností - pomáhá vyhnout se cyklům
    if random.random() < 0.9:  # 90% času vybíráme podle nejvyšší pravděpodobnosti
        move = np.argmax(probs)
    else:  # 10% času vybíráme náhodně pro průzkum
        move = random.choices(range(4), weights=probs)[0]

    # Pokus o pohyb v daném směru
    new_x = position[0] + directions[move][0]
    new_y = position[1] + directions[move][1]

    # Kontrola, zda je nová pozice platná
    if 0 <= new_y < len(maze) and 0 <= new_x < len(maze[0]) and maze[new_y][new_x] != '#':
        return (new_x, new_y)

    # Pokud je směr blokován, zkusíme ostatní směry podle jejich pravděpodobností
    backup_moves = list(range(4))
    backup_moves.remove(move)
    random.shuffle(backup_moves)  # Náhodné pořadí záložních pohybů

    for backup_move in backup_moves:
        new_x = position[0] + directions[backup_move][0]
        new_y = position[1] + directions[backup_move][1]
        if 0 <= new_y < len(maze) and 0 <= new_x < len(maze[0]) and maze[new_y][new_x] != '#':
            return (new_x, new_y)

    return position  # Pokud všechny směry blokované, zůstaneme stát

# ---------- Vizualizace ----------
def print_maze(agent_pos, goal_pos, visited=None):
    os.system('cls' if os.name == 'nt' else 'clear')

    if visited is None:
        visited = set()

    for y, row in enumerate(maze):
        row_str = ''
        for x, cell in enumerate(row):
            if (x, y) == agent_pos:
                row_str += 'A' # agent
            elif (x, y) == goal_pos:
                row_str += 'G' # cíl
            elif (x, y) in visited and cell != '#': # stěny
                row_str += '.' # navštívené buňky
            else:
                row_str += cell
        print(row_str)
    time.sleep(0.2)

# ---------- Simulace jednoho běhu agenta ----------
def simulate_agent(genome, max_steps=100, visualize=False):
    # Reset paměti při každém novém běhu
    get_sensor_inputs.memory = [start_pos] * 3

    pos = start_pos
    path = [pos]
    visited = {pos}
    steps = 0
    visited_count = 1

    # Pro detekci cyklů
    cycle_detector = {}

    # Algoritmus pro detekci uvíznutí
    stuck_counter = 0
    prev_positions = []

    while steps < max_steps:
        if visualize:
            print_maze(pos, goal_pos, visited)
            print(f"Krok: {steps+1}/{max_steps}, Navštíveno buněk: {visited_count}")

        # Detekce cyklů
        state_key = (pos, tuple(sorted(visited)))
        if state_key in cycle_detector:
            stuck_counter += 1
            if stuck_counter > 10:  # Pokud uvízneme v cyklu na 10 kroků
                if visualize:
                    print("Agent uvízl v cyklu!")
                break
        else:
            cycle_detector[state_key] = steps
            stuck_counter = 0

        # Kontrola pro uvíznutí na místě
        prev_positions.append(pos)
        if len(prev_positions) > 5:
            prev_positions.pop(0)
            if all(p == prev_positions[0] for p in prev_positions):
                if visualize:
                    print("Agent uvízl na místě!")
                break

        # Pohyb agenta
        new_pos = nn_navigate_me(pos, genome)
        steps += 1

        # Pokud se pozice změnila a nová pozice nebyla navštívena
        if new_pos != pos and new_pos not in visited:
            visited_count += 1
            visited.add(new_pos)

        pos = new_pos
        path.append(pos)

        # Kontrola dosažení cíle
        if pos == goal_pos:
            if visualize:
                print_maze(pos, goal_pos, visited)
                print(f"Cíl dosažen za {steps} kroků!")
                print(f"Efektivita: {len(path) / max(visited_count, 1):.2f}")
            return True, steps, visited_count, path

    # Pokud jsme nedosáhli cíle
    if visualize:
        print_maze(pos, goal_pos, visited)
        print("Cíl nebyl dosažen v daném počtu kroků.")

    # Výpočet nejkratší vzdálenosti k cíli
    min_dist_to_goal = float('inf')
    for p in path:
        dist = abs(p[0] - goal_pos[0]) + abs(p[1] - goal_pos[1])  # Manhattan distance
        min_dist_to_goal = min(min_dist_to_goal, dist)

    return False, steps, visited_count, path, min_dist_to_goal

# ---------- Vylepšená fitness funkce ----------
def evaluate(genome):
    trials = 5
    success_count = 0
    total_steps = 0
    total_visited = 0
    min_distances = []
    path_progress_scores = []
    unique_cell_ratios = []
    cycle_avoidance_scores = []

    for _ in range(trials):
        result = simulate_agent(genome)

        if result[0]:  # Úspěšný běh
            success, steps, visited, path = result
            success_count += 1
            total_steps += steps
            total_visited += visited

            # Efektivita cesty - poměr délky cesty k počtu navštívených buněk
            path_efficiency = len(set(path)) / len(path) if len(path) > 0 else 0
            unique_cell_ratios.append(path_efficiency)

            # Bonus za rychlé dosažení cíle
            time_bonus = max(0, 1 - (steps / 50))  # Bonus pokud dosáhne cíle do 50 kroků
            path_progress_scores.append(1.0 + time_bonus)

            # Dobrá schopnost vyhýbat se cyklům
            cycle_avoidance_scores.append(1.0)
        else:  # Neúspěšný běh
            success, steps, visited, path, min_dist = result
            total_steps += steps
            total_visited += visited
            min_distances.append(min_dist)

            # Měření postupného přibližování k cíli
            distances_to_goal = []
            for p in path:
                dist = abs(p[0] - goal_pos[0]) + abs(p[1] - goal_pos[1])  # Manhattan distance
                distances_to_goal.append(dist)

            # Výpočet průměrné vzdálenosti v první a druhé polovině cesty
            half = len(distances_to_goal) // 2
            if half > 0:
                first_half_avg = sum(distances_to_goal[:half]) / half
                second_half_avg = sum(distances_to_goal[half:]) / (len(distances_to_goal) - half)
                # Pokud se agent přibližuje k cíli, druhá polovina by měla mít menší průměrnou vzdálenost
                progress = max(0, (first_half_avg - second_half_avg) / first_half_avg) if first_half_avg > 0 else 0
                path_progress_scores.append(progress)
            else:
                path_progress_scores.append(0)

            # Poměr unikátních navštívených buněk k celkovému počtu kroků
            unique_ratio = len(set(path)) / len(path) if len(path) > 0 else 0
            unique_cell_ratios.append(unique_ratio)

            # Detekce cyklů - penalizace za uvíznutí v cyklech
            # Jednoduchá metrika: pokud agent navštívil méně než 50% unikátních buněk z celkového počtu kroků
            cycle_score = unique_ratio * 2  # 0 až 2, kde 1 je neutrální (50% unikátních buněk)
            cycle_avoidance_scores.append(min(cycle_score, 1.0))  # Omezení na max 1.0

    # Základní fitness - úspěšnost (zvýšená váha)
    success_rate = success_count / trials

    # Průměrný počet kroků (normalizovaný)
    avg_steps = total_steps / (trials * 100) if trials > 0 else 1  # max_steps je 100
    step_efficiency = 1 - min(avg_steps, 1)  # Menší počet kroků = vyšší efektivita

    # Průměrný počet navštívených buněk
    avg_visited = total_visited / trials if trials > 0 else 0
    exploration_score = min(avg_visited / 15, 1)  # Normalizace - 15 je přibližně počet dostupných buněk

    # Pro neúspěšné běhy - jak blízko se agent dostal k cíli
    distance_score = 0
    if min_distances:
        # Průměrná minimální vzdálenost k cíli (nižší = lepší)
        avg_min_dist = sum(min_distances) / len(min_distances)
        distance_score = 1 - min(avg_min_dist / max_distance, 1)

    # Nové metriky
    path_progress = sum(path_progress_scores) / len(path_progress_scores) if path_progress_scores else 0
    unique_cell_ratio = sum(unique_cell_ratios) / len(unique_cell_ratios) if unique_cell_ratios else 0
    cycle_avoidance = sum(cycle_avoidance_scores) / len(cycle_avoidance_scores) if cycle_avoidance_scores else 0

    # Skládání fitness funkcí s různými váhami - upravené váhy a nové komponenty
    fitness = (
            0.4 * success_rate +             # Hlavní cíl - dosáhnout cíle (snížená váha)
            0.15 * step_efficiency +         # Efektivita kroků
            0.05 * exploration_score +       # Průzkum bludiště (snížená váha)
            0.15 * distance_score +          # Blízkost k cíli pro neúspěšné běhy (zvýšená váha)
            0.1 * path_progress +            # Nová metrika - postupné přibližování k cíli
            0.1 * unique_cell_ratio +        # Nová metrika - efektivita cesty (poměr unikátních buněk)
            0.05 * cycle_avoidance           # Nová metrika - schopnost vyhnout se cyklům
    )

    return fitness,

# ---------- DEAP konfigurace ----------
# Vyčistit již definované třídy
if 'FitnessMax' in dir(creator):
    del creator.FitnessMax
if 'Individual' in dir(creator):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Velikost genomu pro neuronovou síť s 3 vrstvami
input_size = 10  # 4 vzdálenosti + 2 souřadnice (vzdálenost a úhel k cíli) + 1 křižovatka + 3 předchozí pohyby (pouze x)
hidden1_size = 12
hidden2_size = 8
output_size = 4

# Celková velikost genomu
genome_size = (input_size * hidden1_size) + hidden1_size + (hidden1_size * hidden2_size) + hidden2_size + (hidden2_size * output_size) + output_size

# Inicializace genomu s rozsahem HE inicializace pro vrstvy
def init_layer_he(size_in, size_out):
    std = np.sqrt(2 / size_in)
    return lambda: random.normalvariate(0, std)

toolbox.register("gene_w1", init_layer_he(input_size, hidden1_size))
toolbox.register("gene_b1", init_layer_he(1, hidden1_size))
toolbox.register("gene_w2", init_layer_he(hidden1_size, hidden2_size))
toolbox.register("gene_b2", init_layer_he(1, hidden2_size))
toolbox.register("gene_w3", init_layer_he(hidden2_size, output_size))
toolbox.register("gene_b3", init_layer_he(1, output_size))

# Vytvoření jednotlivce s odpovídajícími geny pro každou vrstvu
def init_individual():
    w1_size = input_size * hidden1_size
    b1_size = hidden1_size
    w2_size = hidden1_size * hidden2_size
    b2_size = hidden2_size
    w3_size = hidden2_size * output_size
    b3_size = output_size

    genes = []
    # Váhy a biasy pro první vrstvu
    genes.extend([toolbox.gene_w1() for _ in range(w1_size)])
    genes.extend([toolbox.gene_b1() for _ in range(b1_size)])
    # Váhy a biasy pro druhou vrstvu
    genes.extend([toolbox.gene_w2() for _ in range(w2_size)])
    genes.extend([toolbox.gene_b2() for _ in range(b2_size)])
    # Váhy a biasy pro výstupní vrstvu
    genes.extend([toolbox.gene_w3() for _ in range(w3_size)])
    genes.extend([toolbox.gene_b3() for _ in range(b3_size)])

    return creator.Individual(genes)

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Vylepšené operátory
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.1)  # Blend crossover místo simple crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.15)  # Jemnější mutace
toolbox.register("select", tools.selTournament, tournsize=5)  # Větší turnaj pro silnější selekční tlak

# ---------- Vylepšený tréninkový proces ----------
def train(pop_size=100, ngen=75, target=0.95):
    # Inicializace populace
    pop = toolbox.population(n=pop_size)

    # Vyhodnocení počáteční populace
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Hall of Fame pro uchování nejlepších jedinců
    hof = tools.HallOfFame(5)  # Uchováme 5 nejlepších genomů
    hof.update(pop)

    # Statistiky pro sledování evoluce
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    # Parametry pro postupné snižování šance mutace
    initial_mutpb = 0.3
    final_mutpb = 0.1

    # Elitismus - počet nejlepších jedinců, kteří přežijí beze změny
    elite_size = 5

    # Evoluce
    for gen in range(ngen):
        # Adaptivní parametry - postupné snižování míry mutace
        mutpb = initial_mutpb - (initial_mutpb - final_mutpb) * (gen / ngen)

        # Elitismus - uložení nejlepších jedinců
        elites = tools.selBest(pop, elite_size)
        elites = list(map(toolbox.clone, elites))

        # Selekce a vytvoření nové generace
        offspring = toolbox.select(pop, len(pop) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # Aplikace crossover a mutace na potomky
        for i in range(1, len(offspring), 2):
            if i < len(offspring) - 1:  # Kontrola, zda máme pár jedinců
                if random.random() < 0.7:  # Pravděpodobnost crossoveru
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:  # Adaptivní pravděpodobnost mutace
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Vyhodnocení jedinců s neznámou fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Nahrazení populace s elitismem
        pop[:] = elites + offspring

        # Aktualizace Hall of Fame
        hof.update(pop)

        # Výpis statistik
        record = stats.compile(pop)
        print(f"Generace {gen+1}/{ngen}: min = {record['min']:.3f}, avg = {record['avg']:.3f}, max = {record['max']:.3f}, std = {record['std']:.3f}")

        # Kontrola dosažení cíle
        if hof[0].fitness.values[0] >= target:
            print(f"✅ Cíl dosažen v generaci {gen+1} – agent dosahuje cíle s vysokou úspěšností!")
            break

        # Restart nejhoršího jedince jako kopii nejlepšího s mutací, pokud dojde k stagnaci
        if gen > 10 and record['std'] < 0.01:  # Detekce stagnace
            worst_idx = np.argmin([ind.fitness.values[0] for ind in pop])
            pop[worst_idx] = toolbox.clone(hof[0])
            toolbox.mutate(pop[worst_idx])
            del pop[worst_idx].fitness.values
            print("Detekována stagnace - restart nejhoršího jedince")

    return hof[0]

# ---------- Spusť trénink ----------
print("Začíná trénink vylepšeného agenta...")
best_genome = train(pop_size=150, ngen=100)

# ---------- Test nejlepšího genomu ----------
print("\nTest nejlepšího genomu:")
successes = 0
total_steps = 0
for i in range(20):
    result = simulate_agent(best_genome, max_steps=100, visualize=False)
    if result[0]:
        successes += 1
        total_steps += result[1]

avg_steps = total_steps / max(1, successes)
print(f"Úspěšnost: {successes}/20 ({successes * 5}%)")
if successes > 0:
    print(f"Průměrný počet kroků pro úspěšné běhy: {avg_steps:.1f}")

# ---------- Vizualizuj jeden úspěšný běh ----------
print("\n Vizualizace běhu nejlepšího agenta:")
simulate_agent(best_genome, max_steps=100, visualize=True)

# Uložení nejlepšího genomu pro budoucí použití
try:
    np.save("best_maze_agent.npy", np.array(best_genome))
    print("\n Nejlepší genom byl uložen jako 'best_maze_agent.npy'")
except:
    print("\n Nepodařilo se uložit genom")
