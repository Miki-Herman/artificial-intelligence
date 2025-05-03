import numpy as np
import random

# Bludiště jako textová mapa
MAZE = [
    "#########",
    "#     #G#",
    "# ### # #",
    "# #     #",
    "# ##### #",
    "#       #",
    "#########"
]

# Konverze na 2D pole
maze = [list(row) for row in MAZE]

# Najdi start a cíl
start_pos = (1, 1)
goal_pos = (7, 1)

# Agent
class Agent:
    def __init__(self, pos, genome=None):
        self.position = pos
        self.genome = genome if genome is not None else np.random.randn(6, 4)
        self.steps = 0

# Senzory – okolní stěny a směr k cíli
def get_sensor_inputs(agent, maze, goal):
    x, y = agent.position
    walls = [
        int(maze[y-1][x] != '#'),  # nahoru
        int(maze[y+1][x] != '#'),  # dolů
        int(maze[y][x-1] != '#'),  # vlevo
        int(maze[y][x+1] != '#')   # vpravo
    ]
    dx = (goal[0] - x) / len(maze[0])
    dy = (goal[1] - y) / len(maze)
    return walls + [dx, dy]

# Neuronová síť (jednoduchá lineární vrstva + tanh)
def nn_function(inp, wei):
    output = np.dot(inp, wei)
    return np.tanh(output)

# Navigace agenta
def nn_navigate_me(agent, maze, goal):
    inp = get_sensor_inputs(agent, maze, goal)
    out = nn_function(inp, agent.genome)
    move = np.argmax(out)  # 0=up, 1=down, 2=left, 3=right
    x, y = agent.position
    new_pos = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)][move]
    if maze[new_pos[1]][new_pos[0]] != '#':
        agent.position = new_pos
    agent.steps += 1

# Simulace 1 agenta
agent = Agent(start_pos)

for _ in range(50):  # max. počet kroků
    nn_navigate_me(agent, maze, goal_pos)
    if agent.position == goal_pos:
        print("Agent dosáhl cíle za", agent.steps, "kroků!")
        break
else:
    print("Agent cíl nenašel. Poslední pozice:", agent.position)
