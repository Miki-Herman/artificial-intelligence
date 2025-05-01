import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

"""
Definitions: 
1. Peak is when neighbouring points are lower than the 'peak' point
2. Water level is constant on generated terrain and its 0.5
3. Lake is considered part of terrain lower than 0.5 and its lowest point is 0.4
4. Sea is considered part of terrain lower than 0.5

Rules affecting evolution:
1. Number of peaks 
2. Number of lakes 
3. Number of seas
4. Underwater percentage
5. Variability (No extremes, and not too much flat terrains)
6. Terrain roughness (Smoother or rougher terrain)
"""

def plotterrain(t):
    """
    Terrain is a list of float numbers between 0 and 1, and water level is 0.5

    :param t: list of float numbers representing the terrain
    :return: None
    """

    fig, ax = plt.subplots()

    x = range(len(t))
    sea = [0.5 for i in range(len(t))]

    ax.fill_between(x, sea, color="turquoise")
    ax.fill_between(x, t, color="sandybrown")

    plt.show()


def count_peaks(t):
    """
    Used to count the number of peaks in the terrain.
    The definition of a peak is on the line #6

    :param t: list of floats that represents points in the terrain
    :return: number of peaks in the terrain
    """
    count = 0
    for i in range(len(t)):

        # edge case 1 --> first point in terrain
        if i == 0 and t[i] > t[i + 1]:
            count += 1

        # edge case 2 --> last point in terrain
        elif i == len(t) - 1 and t[i] > t[i - 1]:
            count += 1

        # point in terrain
        elif 0 < i <len(t) and t[i-1] < t[i] > t[i+1]:
            count += 1

    return count

def count_lakes_seas(t):
    """
    Used to count the number of lakes and seas in the terrain.
    Definitions of lake and sea in this program is on line #8 and #9

    :param t: list of floats that represents points in the terrain
    :return: number of lakes and seas in the terrain
    """
    lake_count = 0
    sea_count = 0
    in_water = False
    min_height = 1.0

    for height in t:
        if not in_water and height <= 0.5:  # entering water
            in_water = True
            min_height = height  # start tracking minimum height
        elif in_water:
            if height <= 0.5:  # still in water
                min_height = min(min_height, height)  # update minimum height
            else:  # exiting water
                # classify water body based on minimum height
                if min_height < 0.4:
                    sea_count += 1
                else:
                    lake_count += 1
                in_water = False

    # edge case -> terrain ends while still in water
    if in_water:
        if min_height < 0.4:
            sea_count += 1
        else:
            lake_count += 1

    return lake_count, sea_count

def water_coverage(t):
    """
    Used to calculate how much land is underwater

    :param t: list of floats that represents points in the terrain
    :return: percentage of water coverage
    """
    # water level is 0.5
    water_level = 0.5
    return sum(1 for i in t if i < water_level) / len(t)

def terrain_variability(t):
    """
    :param t: list of floats that represents points in the terrain
    :return: Returns the standard deviation of the terrain
    """
    return np.std(t)

def terrain_roughness(t):
    """
    Used to determine the terrain roughness

    :param t: list of floats that represents points in the terrain
    :return: average difference between two neighboring points
    """
    return sum(abs(t[i] - t[i-1]) for i in range(1, len(t))) / (len(t)-1)

def bounded_mutation(individual, mu, sigma, indpb):
    tools.mutGaussian(individual, mu, sigma, indpb)
    for i in range(len(individual)):
        individual[i] = min(max(individual[i], 0.0), 1.0)
    return individual,

def evaluate(terrain):
    peaks = count_peaks(terrain)
    lakes, seas = count_lakes_seas(terrain)
    coverage = water_coverage(terrain)
    variability = terrain_variability(terrain)
    roughness = terrain_roughness(terrain)

    # punish high peaks
    peak_penality = sum(1 for h in terrain if h > 0.8) * -0.5

    # Goals: 5-6 peaks, 2–3 lakes, 0-1 seas, 40% water, moderate variability, smoothness
    peak_score = -abs(peaks - 3.5) # best if ~5-6 peaks
    lake_score = -abs(lakes - 2.5)  # best if ~6–7 lakes
    sea_score = -abs(seas - 1.5) # best if ~0–1 seas
    water_score = -abs(coverage - 0.6)  # best if ~40% flooded
    var_score = -abs(variability - 0.08)  # ideal variability
    rough_score = -abs(roughness - 0.02)   # ideal roughness

    return peak_score + lake_score + sea_score + water_score + var_score + rough_score + peak_penality,

def main():
    LENGTH = 50     # number of terrain points
    POP_SIZE = 1000  # number of individuals in the population
    NGEN = 100       # number of generations to evolve
    MUTPB = 0.2      # probability of mutation
    CXPB = 0.5       # probability of crossover

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.1, 0.9)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", bounded_mutation, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=15)

    pop = toolbox.population(n=POP_SIZE)

    # evolution
    for gen in range(NGEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

        top = tools.selBest(pop, 1)[0]
        print(f"Gen {gen} Best fitness: {top.fitness.values[0]:.4f}")

    # show the best terrain
    best = tools.selBest(pop, 1)[0]
    print(best)
    plotterrain(best)

if __name__ == "__main__":
    main()

