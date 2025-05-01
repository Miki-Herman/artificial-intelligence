import random
import time
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, path: str):
        """Initializes the graph by reading a DIMACS file."""
        self.graph_name = f"{path.split('/')[-1].split('.')[0]}.pdf"
        self.graph = self.load_dimacs(path)

    def load_dimacs(self, path):
        """Parses a DIMACS .col file and builds an adjacency list."""
        graph = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "e":
                    u, v = int(parts[1]), int(parts[2])
                    graph.setdefault(u, set()).add(v)
                    graph.setdefault(v, set()).add(u)
        return graph

    def is_coloring(self, coloring):
        """Checks if the given coloring is valid (no adjacent nodes share the same color)."""
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                if coloring.get(node) == coloring.get(neighbor):
                    return False
        return True

    def node_conflicts(self, node, coloring):
        """Returns number of conflicts for a specific node."""
        return sum(1 for neighbor in self.graph[node] if coloring[node] == coloring[neighbor])

    def count_conflicts(self, coloring):
        """Total number of color conflicts in the entire graph."""
        conflicts = 0
        for node in self.graph:
            for neighbor in self.graph[node]:
                if coloring[node] == coloring[neighbor]:
                    conflicts += 1
        return conflicts // 2  # Each conflict is counted twice

    def color(self, k, steps):
        """
        Attempts to color the graph using k colors with optimized hill climbing.
        Returns (coloring_dict, solved_bool)
        """
        nodes = list(self.graph.keys())
        coloring = {node: random.randint(0, k - 1) for node in nodes}

        for step in range(steps):
            # Find conflicted nodes only
            conflicted_nodes = [node for node in nodes if self.node_conflicts(node, coloring) > 0]
            print(f"Computing step number {step}: number of conflicts {len(conflicted_nodes)}")

            if not conflicted_nodes:
                return coloring, True  # No conflicts left

            node = random.choice(conflicted_nodes)
            current_color = coloring[node]
            best_color = current_color
            min_conflicts = self.node_conflicts(node, coloring)

            for color in range(k):
                coloring[node] = color
                conflicts = self.node_conflicts(node, coloring)
                if conflicts < min_conflicts:
                    best_color = color
                    min_conflicts = conflicts
                    if conflicts == 0:
                        break  # Best possible

            coloring[node] = best_color  # Final decision

            # Optional random walk to escape local optima
            if best_color == current_color and random.random() < 0.1:
                coloring[node] = random.randint(0, k - 1)

        return coloring, False  # Return best effort if steps exceeded

    def visualize(self, coloring, max_colors, time):
        """Visualizes the graph using NetworkX and Matplotlib."""
        G = nx.Graph()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Assign colors to nodes
        unique_colors = list(set(coloring.values()))
        cmap = plt.colormaps.get_cmap("tab10")
        color_map = {color: cmap(i % 10) for i, color in enumerate(unique_colors)}
        node_colors = [color_map[coloring[node]] for node in G.nodes()]

        plt.figure(figsize=(10, 10))
        nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=8)
        plt.savefig(f"output/max_colors_{max_colors}_{time}s_{self.graph_name}", format="pdf", bbox_inches="tight")
        plt.show()


# Example Usage
file_path = "graphs/dsjc125.9.col"
g = Graph(file_path)
i = 0
steps = 1000000000000
max_colors = 44

start_time = time.time()
coloring, solved = g.color(k=max_colors, steps=steps)
end_time = time.time()

in_sec = int(end_time - start_time)

if solved:
    g.visualize(coloring, max_colors, in_sec)
    print(coloring, solved)

else:
    print(f"Failed to solve on iteration in {steps} steps with max {max_colors} colors")