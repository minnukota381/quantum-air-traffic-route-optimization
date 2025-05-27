import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use a backend that does not need GUI support
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Dimensions of the airspace
X_DIM = 5  # east-west
Y_DIM = 5  # north-south
Z_DIM = 3  # altitude levels
T_MAX = 10  # time steps

def create_4d_grid():
    G = nx.DiGraph()  # Directed graph

    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    node = (x, y, z, t)
                    G.add_node(node)

                    # Possible movements
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,1), (0,0,-1), (0,0,0)]:
                        nx_ = x + dx
                        ny_ = y + dy
                        nz_ = z + dz
                        if 0 <= nx_ < X_DIM and 0 <= ny_ < Y_DIM and 0 <= nz_ < Z_DIM:
                            next_node = (nx_, ny_, nz_, t+1)
                            if t+1 < T_MAX:
                                G.add_edge(node, next_node)
    return G

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

def find_paths(graph, flights):
    flight_paths = {}
    for flight in flights:
        try:
            path = nx.shortest_path(graph, source=flight['start'], target=flight['end'])
            flight_paths[flight['id']] = path
            print(f"Flight {flight['id']} path found with {len(path)} steps.")
        except nx.NetworkXNoPath:
            print(f"No path found for flight {flight['id']}")
            flight_paths[flight['id']] = None
    return flight_paths

def visualize_flight_paths_3d(flight_paths):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'b', 'g', 'm', 'c', 'y']
    for i, (flight_id, path) in enumerate(flight_paths.items()):
        if path is None:
            continue
        xs = [node[0] for node in path]
        ys = [node[1] for node in path]
        zs = [node[2] for node in path]
        ax.plot(xs, ys, zs, marker='o', label=flight_id, color=colors[i % len(colors)])

    ax.set_xlabel('X (Longitude)')
    ax.set_ylabel('Y (Latitude)')
    ax.set_zlabel('Z (Altitude)')
    ax.set_title('3D Flight Paths')
    ax.legend()
    plt.savefig("./images/flight_paths_3d.png")
    plt.close()
    print("3D flight paths image saved as ./images/flight_paths_3d.png")

if __name__ == "__main__":
    grid = create_4d_grid()
    print(f"Total Nodes: {len(grid.nodes)}")
    print(f"Total Edges: {len(grid.edges)}")

    flights = define_flights()
    flight_paths = find_paths(grid, flights)

    # Optional: print full paths step-by-step
    for fid, path in flight_paths.items():
        if path:
            print(f"\nFull path for Flight {fid}:")
            for node in path:
                print(f"  Position (x,y,z)={node[:3]} at time t={node[3]}")

    visualize_flight_paths_3d(flight_paths)
