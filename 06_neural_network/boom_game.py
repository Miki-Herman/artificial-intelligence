# -*- coding: utf-8 -*-

import pygame
import random
import numpy as np
import math
import copy

# Constants for the game
WIDTH, HEIGHT = 900, 500
FPS = 80
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize pygame
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boom master - Neural Network Evolution")
pygame.font.init()

# Load fonts
BOOM_FONT = pygame.font.SysFont("comicsans", 100)
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)

# Load images
ENEMY_IMAGE = pygame.image.load("mine.png")
ME_IMAGE = pygame.image.load("me.png")
SEA_IMAGE = pygame.image.load("sea.png")
FLAG_IMAGE = pygame.image.load("flag.png")

# Size of game entities
ENEMY_SIZE = 50
ME_SIZE = 50

# Transform images to the correct size
ENEMY = pygame.transform.scale(ENEMY_IMAGE, (ENEMY_SIZE, ENEMY_SIZE))
ME = pygame.transform.scale(ME_IMAGE, (ME_SIZE, ME_SIZE))
SEA = pygame.transform.scale(SEA_IMAGE, (WIDTH, HEIGHT))
FLAG = pygame.transform.scale(FLAG_IMAGE, (ME_SIZE, ME_SIZE))

# Custom events
ME_HIT = pygame.USEREVENT + 1
ME_WIN = pygame.USEREVENT + 2

# Flag position
FLAG_POS = (WIDTH - ME_SIZE, HEIGHT - ME_SIZE - 10)

# Evolution parameters
POPULATION_SIZE = 30
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.3
ELITE_SIZE = 5
MAX_GENERATIONS = 100
SIMULATION_STEPS = 2000  # Maximum steps in one simulation
NUM_INPUTS = 10  # Number of sensory inputs
NUM_OUTPUTS = 4  # Up, Down, Left, Right
NUM_HIDDEN = 8   # Number of hidden neurons

# Velocity for agents
VELOCITY = 5

# Mine class
class Mine:
    def __init__(self):
        # Random x direction
        if random.random() > 0.5:
            self.dirx = 1
        else:
            self.dirx = -1

        # Random y direction
        if random.random() > 0.5:
            self.diry = 1
        else:
            self.diry = -1

        x = random.randint(200, WIDTH - ENEMY_SIZE)
        y = random.randint(200, HEIGHT - ENEMY_SIZE)
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)

        self.velocity = random.randint(1, 5)

# Agent (me) class with neural network
class Agent:
    def __init__(self, genome=None):
        self.rect = pygame.Rect(10, 10, ME_SIZE, ME_SIZE)
        self.fitness = 0
        self.steps_alive = 0
        self.reached_flag = False
        self.dead = False

        # Initialize neural network weights
        if genome is None:
            # Input to hidden layer weights
            self.weights1 = np.random.randn(NUM_INPUTS, NUM_HIDDEN)
            # Hidden to output layer weights
            self.weights2 = np.random.randn(NUM_HIDDEN, NUM_OUTPUTS)
        else:
            self.weights1 = genome[0].copy()
            self.weights2 = genome[1].copy()

    def get_genome(self):
        return [self.weights1, self.weights2]

    def calc_fitness(self):
        # Base fitness is how many steps the agent survived
        self.fitness = self.steps_alive

        if self.reached_flag:
            # Big bonus for reaching the flag
            self.fitness += 10000
            # Additional bonus for reaching it quickly
            self.fitness += (5000 - self.steps_alive)
        else:
            # Factor in distance to the flag if didn't reach it
            dist_to_flag = math.sqrt((self.rect.x - FLAG_POS[0])**2 +
                                     (self.rect.y - FLAG_POS[1])**2)
            # Closer to flag is better (inversely proportional)
            self.fitness += (WIDTH + HEIGHT - dist_to_flag) * 5

        return self.fitness

# Function to create a set number of mines
def set_mines(num):
    mines = []
    for i in range(num):
        m = Mine()
        mines.append(m)
    return mines

# 1. Implement sensory functions for agents
def get_agent_sensors(agent, mines):
    """
    Get sensory inputs for the agent. Returns a list of inputs:
    - Distance to closest mine in 8 directions (normalized)
    - Distance to flag (normalized)
    - Flag direction angle (normalized)
    """
    inputs = np.zeros(NUM_INPUTS)

    # 1-8: Distance to nearest mine in 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = [
        (0, -1),    # North
        (1, -1),    # Northeast
        (1, 0),     # East
        (1, 1),     # Southeast
        (0, 1),     # South
        (-1, 1),    # Southwest
        (-1, 0),    # West
        (-1, -1)    # Northwest
    ]

    for i, (dx, dy) in enumerate(directions):
        # Start with maximum possible distance
        min_dist = math.sqrt(WIDTH**2 + HEIGHT**2)

        for mine in mines:
            # Check if mine is in this direction
            if dx != 0:
                if (mine.rect.x - agent.rect.x) * dx < 0:
                    continue
            if dy != 0:
                if (mine.rect.y - agent.rect.y) * dy < 0:
                    continue

            # Calculate distance to the mine
            dist = math.sqrt((mine.rect.x - agent.rect.x)**2 +
                             (mine.rect.y - agent.rect.y)**2)

            # Check if it's in the right direction
            if dx != 0 and dy != 0:  # Diagonal
                # Calculate angle to the mine
                angle = math.atan2(mine.rect.y - agent.rect.y,
                                   mine.rect.x - agent.rect.x)
                target_angle = math.atan2(dy, dx)
                angle_diff = abs(angle - target_angle)

                # If the angle is close enough, consider it in this direction
                if angle_diff < math.pi/4:
                    min_dist = min(min_dist, dist)
            else:  # Cardinal direction
                if dx == 0:  # North or South
                    if abs(mine.rect.x - agent.rect.x) < ENEMY_SIZE:
                        min_dist = min(min_dist, dist)
                elif dy == 0:  # East or West
                    if abs(mine.rect.y - agent.rect.y) < ENEMY_SIZE:
                        min_dist = min(min_dist, dist)

        # Normalize distance (invert so closer = higher value)
        if min_dist == math.sqrt(WIDTH**2 + HEIGHT**2):
            inputs[i] = 0  # No mine in this direction
        else:
            inputs[i] = 1 - (min_dist / math.sqrt(WIDTH**2 + HEIGHT**2))

    # 9: Distance to flag (normalized)
    dist_to_flag = math.sqrt((agent.rect.x - FLAG_POS[0])**2 +
                             (agent.rect.y - FLAG_POS[1])**2)
    inputs[8] = 1 - (dist_to_flag / math.sqrt(WIDTH**2 + HEIGHT**2))

    # 10: Angle to flag (normalized to [-1, 1])
    angle_to_flag = math.atan2(FLAG_POS[1] - agent.rect.y,
                               FLAG_POS[0] - agent.rect.x)
    inputs[9] = angle_to_flag / math.pi

    return inputs

# 2. Neural network function
def nn_function(inp, weights):
    """
    Process inputs through the neural network with given weights.

    Args:
        inp: Input vector (sensory inputs)
        weights: List of weight matrices [w1, w2] where:
            - w1: Weights from input to hidden layer
            - w2: Weights from hidden to output layer

    Returns:
        Output vector (movement decisions)
    """
    # First layer
    hidden = np.tanh(np.dot(inp, weights[0]))
    # Output layer
    output = np.tanh(np.dot(hidden, weights[1]))

    return output

# 3. Neural network navigation function
def nn_navigate_me(agent, inp, mines):
    """
    Navigate the agent based on neural network outputs

    Args:
        agent: The agent to navigate
        inp: Sensory inputs

    Returns:
        None (updates agent's position directly)
    """
    # Get weights from agent's genome
    weights = [agent.weights1, agent.weights2]

    # Get neural network outputs
    outputs = nn_function(inp, weights)

    # Decide movement based on outputs [up, down, left, right]
    move_x = 0
    move_y = 0

    # Each output controls one direction
    if outputs[0] > 0.5:  # Up
        move_y -= VELOCITY
    if outputs[1] > 0.5:  # Down
        move_y += VELOCITY
    if outputs[2] > 0.5:  # Left
        move_x -= VELOCITY
    if outputs[3] > 0.5:  # Right
        move_x += VELOCITY

    # Update agent position
    new_x = agent.rect.x + move_x
    new_y = agent.rect.y + move_y

    # Check boundaries
    if new_x < 0:
        new_x = 0
    elif new_x > WIDTH - agent.rect.width:
        new_x = WIDTH - agent.rect.width

    if new_y < 0:
        new_y = 0
    elif new_y > HEIGHT - agent.rect.height:
        new_y = HEIGHT - agent.rect.height

    # Update agent position
    agent.rect.x = new_x
    agent.rect.y = new_y

# 4. Fitness calculation function
def handle_mes_fitnesses(agents, mines, steps_elapsed):
    """
    Calculate fitness for each agent.

    Args:
        agents: List of agents
        mines: List of mines
        steps_elapsed: Number of steps elapsed in the simulation

    Returns:
        List of updated agents with calculated fitness
    """
    for agent in agents:
        if not agent.dead:
            # Increment steps alive
            agent.steps_alive = steps_elapsed

            # Check if agent has reached the flag
            if agent.rect.x > WIDTH - ME_SIZE - 15 and agent.rect.y > HEIGHT - ME_SIZE - 15:
                agent.reached_flag = True
                agent.dead = True  # Mark as done

            # Check if agent has collided with a mine
            for mine in mines:
                if agent.rect.colliderect(mine.rect):
                    agent.dead = True
                    break

        # Calculate fitness for this agent
        agent.calc_fitness()

    return agents

# Evolutionary operations
def selection(population):
    """Tournament selection"""
    selected = []

    # Keep elite individuals
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    selected.extend(copy.deepcopy(sorted_pop[:ELITE_SIZE]))

    # Tournament selection for the rest
    while len(selected) < POPULATION_SIZE:
        tournament_size = 3
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        selected.append(copy.deepcopy(winner))

    return selected

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    child = Agent()

    # Crossover for weights1
    rows, cols = parent1.weights1.shape
    crossover_point_row = random.randint(0, rows-1)
    crossover_point_col = random.randint(0, cols-1)

    for i in range(rows):
        for j in range(cols):
            if i < crossover_point_row or (i == crossover_point_row and j <= crossover_point_col):
                child.weights1[i][j] = parent1.weights1[i][j]
            else:
                child.weights1[i][j] = parent2.weights1[i][j]

    # Crossover for weights2
    rows, cols = parent1.weights2.shape
    crossover_point_row = random.randint(0, rows-1)
    crossover_point_col = random.randint(0, cols-1)

    for i in range(rows):
        for j in range(cols):
            if i < crossover_point_row or (i == crossover_point_row and j <= crossover_point_col):
                child.weights2[i][j] = parent1.weights2[i][j]
            else:
                child.weights2[i][j] = parent2.weights2[i][j]

    return child

def mutation(agent):
    """Mutate an agent's genome"""
    # Mutate weights1
    rows, cols = agent.weights1.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < MUTATION_RATE:
                agent.weights1[i][j] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)

    # Mutate weights2
    rows, cols = agent.weights2.shape
    for i in range(rows):
        for j in range(cols):
            if random.random() < MUTATION_RATE:
                agent.weights2[i][j] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)

    return agent

def create_next_generation(population):
    """Create the next generation through selection, crossover and mutation"""
    # Selection
    selected = selection(population)

    # Create new population through crossover and mutation
    new_population = []

    # Keep elite individuals unchanged
    new_population.extend(selected[:ELITE_SIZE])

    # Create the rest through crossover and mutation
    while len(new_population) < POPULATION_SIZE:
        parent1 = random.choice(selected)
        parent2 = random.choice(selected)

        if random.random() < 0.7:  # Crossover probability
            child = crossover(parent1, parent2)
        else:
            child = copy.deepcopy(parent1)

        # Apply mutation
        child = mutation(child)

        new_population.append(child)

    return new_population

def handle_mine_movement(mine):
    """Update mine position based on its direction and velocity"""
    if mine.dirx == -1 and mine.rect.x - mine.velocity < 0:
        mine.dirx = 1

    if mine.dirx == 1 and mine.rect.x + mine.rect.width + mine.velocity > WIDTH:
        mine.dirx = -1

    if mine.diry == -1 and mine.rect.y - mine.velocity < 0:
        mine.diry = 1

    if mine.diry == 1 and mine.rect.y + mine.rect.height + mine.velocity > HEIGHT:
        mine.diry = -1

    mine.rect.x += mine.dirx * mine.velocity
    mine.rect.y += mine.diry * mine.velocity

def draw_window(agents, mines, generation, step, best_fitness):
    """Draw the game state"""
    WIN.blit(SEA, (0, 0))

    # Draw flag
    WIN.blit(FLAG, FLAG_POS)

    # Draw text info
    generation_text = LEVEL_FONT.render(f"Generation: {generation}", 1, WHITE)
    step_text = LEVEL_FONT.render(f"Step: {step}", 1, WHITE)
    fitness_text = LEVEL_FONT.render(f"Best Fitness: {best_fitness}", 1, WHITE)
    alive_text = LEVEL_FONT.render(f"Alive: {sum(1 for a in agents if not a.dead)}/{len(agents)}", 1, WHITE)

    WIN.blit(generation_text, (10, HEIGHT - 100))
    WIN.blit(step_text, (10, HEIGHT - 75))
    WIN.blit(fitness_text, (10, HEIGHT - 50))
    WIN.blit(alive_text, (10, HEIGHT - 25))

    # Draw mines
    for mine in mines:
        WIN.blit(ENEMY, (mine.rect.x, mine.rect.y))

    # Draw agents
    for agent in agents:
        if not agent.dead:
            color = GREEN if agent.reached_flag else WHITE
            agent_rect = pygame.Rect(agent.rect.x, agent.rect.y, agent.rect.width, agent.rect.height)
            WIN.blit(ME, (agent.rect.x, agent.rect.y))

    pygame.display.update()

def draw_text(text):
    """Display large text on screen"""
    t = BOOM_FONT.render(text, 1, WHITE)
    WIN.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 - t.get_height() // 2))

    pygame.display.update()

def save_best_genome(agent, generation):
    """Save the genome of the best agent to a file"""
    with open(f"best_genome_gen_{generation}.npy", 'wb') as f:
        np.save(f, agent.weights1)
        np.save(f, agent.weights2)
    print(f"Saved best genome from generation {generation}")

def load_genome(filename):
    """Load a genome from a file"""
    with open(filename, 'rb') as f:
        weights1 = np.load(f)
        weights2 = np.load(f)
    return [weights1, weights2]

def main():
    """Main function to run the evolutionary algorithm"""
    clock = pygame.time.Clock()
    run = True

    # Start with generation 1
    generation = 1
    best_fitness_ever = 0
    best_agent_ever = None

    while run and generation <= MAX_GENERATIONS:
        # Initialize population
        if generation == 1:
            population = [Agent() for _ in range(POPULATION_SIZE)]
        else:
            population = create_next_generation(previous_population)

        # Create mines for this generation
        num_mines = min(2 + generation // 5, 8)  # Slower increase in difficulty
        mines = set_mines(num_mines)

        # Run simulation for this generation
        step = 0
        all_dead = False

        while step < SIMULATION_STEPS and not all_dead and run:
            clock.tick(FPS)

            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Update mine positions
            for mine in mines:
                handle_mine_movement(mine)

            # Navigate each agent
            for agent in population:
                if not agent.dead:
                    # Get sensory inputs
                    inputs = get_agent_sensors(agent, mines)
                    # Navigate agent based on neural network
                    nn_navigate_me(agent, inputs, mines)

            # Calculate fitness and check if agents are dead/reached flag
            handle_mes_fitnesses(population, mines, step)

            # Check if all agents are dead or have reached the flag
            if all(agent.dead for agent in population):
                all_dead = True

            # Draw current state
            best_fitness = max(agent.fitness for agent in population)
            draw_window(population, mines, generation, step, best_fitness)

            step += 1

        # Calculate final fitness for all agents
        handle_mes_fitnesses(population, mines, step)

        # Find best agent in this generation
        best_agent = max(population, key=lambda x: x.fitness)
        best_fitness = best_agent.fitness

        # Update best agent ever if this one is better
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_agent_ever = copy.deepcopy(best_agent)

            # Save best genome every time we find a better one
            save_best_genome(best_agent, generation)

        # Display generation results
        result_text = f"Generation {generation}: Best Fitness = {best_fitness:.2f}"
        print(result_text)
        draw_text(result_text)
        pygame.time.delay(1000)

        # Save population for next generation
        previous_population = population

        # Next generation
        generation += 1

    # Display final results
    if best_agent_ever:
        final_text = f"Training complete! Best fitness: {best_fitness_ever:.2f}"
        print(final_text)
        draw_text(final_text)
        pygame.time.delay(2000)

    pygame.quit()

if __name__ == "__main__":
    main()