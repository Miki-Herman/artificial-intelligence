import random
import time
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SchellingModel:

    """
    A Schelling segregation model is an agent-based model that showcases how people segregate into different groups based on their preferences.

    Rules:
    1. Each cell can be empty or occupied by either type 1 or type 2.
    2. Cell is unhappy when the percentage of neighbouring cells of the same type is lower than the threshold
    3. If cell is unhappy it moves to any empty cell
    """

    def __init__(self, size: int, empty_prob: float, type_prob: float, threshold: float):
        self.size: int = size
        self.empty_prob: float = empty_prob
        self.type_prob: float = type_prob
        self.threshold: float = threshold

    def initialize_grid(self):
        num_of_cells: int = self.size * self.size
        num_of_empty_cells: int = int(num_of_cells * self.empty_prob)
        num_of_type_1_cells: int = int(num_of_cells * self.type_prob)
        num_of_type_2_cells:int = num_of_cells - num_of_empty_cells - num_of_type_1_cells

        cells: list = [0] * num_of_empty_cells + [1] * num_of_type_1_cells + [2] * num_of_type_2_cells
        numpy.random.shuffle(cells)
        grid = numpy.array(cells).reshape(self.size, self.size)
        return num_of_type_1_cells + num_of_type_2_cells, grid


    @staticmethod
    def get_neighbors(grid: numpy.ndarray, x: int, y: int) -> list:
        neighbors = []
        for x_axis in [-1, 0, 1]:
            for y_axis in [-1, 0, 1]:
                if x_axis == 0 and y_axis == 0:
                    continue
                neighbouring_x = x + x_axis
                neighbouring_y = y + y_axis
                if 0 <= neighbouring_x < grid.shape[0] and 0 <= neighbouring_y < grid.shape[1]:
                    neighbors.append(grid[neighbouring_x, neighbouring_y])
        return neighbors


    def is_satisfied(self, grid: numpy.ndarray, x, y) -> bool:
        subject = grid[x, y]

        neighbours = self.get_neighbors(grid, x, y)

        same_type_neighbors = sum(1 for n in neighbours if n == subject)
        total_neighbors = sum(1 for n in neighbours if n != 0)

        return (total_neighbors == 0) or (same_type_neighbors / total_neighbors >= self.threshold)


    def step(self, grid: numpy.ndarray) -> numpy.ndarray:
        size = grid.shape[0]
        unhappy_cells = []
        empty_cells = list(zip(*numpy.where(grid == 0)))

        # locate unhappy cells
        for x in range(size):
            for y in range(size):
                if grid[x,y] != 0 and not self.is_satisfied(grid, x, y):
                    unhappy_cells.append((x,y))

        # shuffle unhappy cells -> for random selection of empty cells
        random.shuffle(unhappy_cells)

        # move unhappy cells to empty cells
        for (x, y) in unhappy_cells:
            if empty_cells:
                new_x, new_y = empty_cells.pop()
                grid[new_x, new_y] = grid[x, y]
                grid[x, y] = 0

        return grid

    @staticmethod
    def plot(grid, step_num):

        colors = ["white", "blue", "red"]
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(3, 3))
        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.title(f'Step {step_num + 1}')
        plt.axis('off')
        plt.show()

    def simulate(self, max_steps=100, speed: int = 0.5):
        total_cells, grid = self.initialize_grid()
        satisfied_counts = []

        for step_num in range(max_steps):
            self.plot(grid, step_num)
            satisfied = sum(self.is_satisfied(grid, x, y) for x in range(self.size) for y in range(self.size) if grid[x, y] != 0)
            satisfied_counts.append(satisfied)

            if satisfied == total_cells:
                break

            new_grid = self.step(grid)
            grid = new_grid

            time.sleep(speed)

        plt.figure(figsize=(8, 4))
        plt.plot(satisfied_counts, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Satisfied agents')
        plt.title('Evolution of Satisfaction')
        plt.show()
