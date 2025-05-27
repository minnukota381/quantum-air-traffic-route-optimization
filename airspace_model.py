import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not need GUI support
import matplotlib.pyplot as plt
import numpy as np

# Dimensions of the airspace
X_DIM = 5  # east-west
Y_DIM = 5  # north-south
Z_DIM = 3  # altitude levels
T_MAX = 10  # time steps

def create_4d_grid():
    G = nx.DiGraph()  # Directed graph: movement from one step to another

    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    node = (x, y, z, t)
                    G.add_node(node)

                    # Define possible movements (up/down/left/right/altitude)
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,1), (0,0,-1), (0,0,0)]:
                        nx_ = x + dx
                        ny_ = y + dy
                        nz_ = z + dz
                        if 0 <= nx_ < X_DIM and 0 <= ny_ < Y_DIM and 0 <= nz_ < Z_DIM:
                            next_node = (nx_, ny_, nz_, t+1)
                            if t+1 < T_MAX:
                                G.add_edge(node, next_node)

    return G


# Sample flight definition
def define_flights():
    return [
        {
            "id": "F1",
            "start": (0, 0, 0, 0),
            "end": (4, 4, 0, 9)
        },
        {
            "id": "F2",
            "start": (4, 0, 1, 0),
            "end": (0, 4, 1, 9)
        }
    ]

# Visualize flight paths (simplified 2D)
def visualize_flights(flights):
    colors = ['r', 'b', 'g', 'm']
    for i, flight in enumerate(flights):
        x = [flight['start'][0], flight['end'][0]]
        y = [flight['start'][1], flight['end'][1]]
        plt.plot(x, y, marker='o', label=flight['id'], color=colors[i % len(colors)])
    plt.title("Flight Paths (2D projection)")
    plt.xlabel("X (Longitude)")
    plt.ylabel("Y (Latitude)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./images/flight_paths.png")  # Save to a PNG image

if __name__ == "__main__":
    grid = create_4d_grid()
    print(f"Total Nodes: {len(grid.nodes)}")
    print(f"Total Edges: {len(grid.edges)}")

    flights = define_flights()
    visualize_flights(flights)
