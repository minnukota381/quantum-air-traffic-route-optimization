import networkx as nx
import matplotlib
matplotlib.use('Agg')  # For environments without GUI, change to 'TkAgg' or remove for local GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Airspace dimensions
X_DIM, Y_DIM, Z_DIM, T_MAX = 5, 5, 3, 10

def create_4d_grid():
    G = nx.DiGraph()
    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    node = (x, y, z, t)
                    G.add_node(node)
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,1),(0,0,-1),(0,0,0)]:
                        nx_ = x + dx
                        ny_ = y + dy
                        nz_ = z + dz
                        if 0 <= nx_ < X_DIM and 0 <= ny_ < Y_DIM and 0 <= nz_ < Z_DIM:
                            next_node = (nx_, ny_, nz_, t+1)
                            if t+1 < T_MAX:
                                G.add_edge(node, next_node)
    return G

def define_flights():
    # For demo, flights move linearly in x,y,z over time steps 0..9
    # Generate intermediate positions as well
    flights = [
        {
            "id": "F1",
            "path": [(x, x, 0, t) for t, x in enumerate(np.linspace(0, 4, T_MAX).astype(int))]
        },
        {
            "id": "F2",
            "path": [(4 - t, t, 1, t) for t in range(T_MAX)]
        }
    ]
    return flights

def animate_flights_3d(flights):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, X_DIM-1)
    ax.set_ylim(0, Y_DIM-1)
    ax.set_zlim(0, Z_DIM-1)

    ax.set_xlabel("X (Longitude)")
    ax.set_ylabel("Y (Latitude)")
    ax.set_zlabel("Z (Altitude)")
    ax.set_title("Flight Paths 3D Animation")

    colors = ['r', 'b', 'g', 'm']
    flight_lines = []
    flight_dots = []

    # Initialize lines and dots for each flight
    for i, flight in enumerate(flights):
        # Start with empty lines/dots
        line, = ax.plot([], [], [], color=colors[i % len(colors)], label=flight["id"])
        dot, = ax.plot([], [], [], marker='o', color=colors[i % len(colors)])
        flight_lines.append(line)
        flight_dots.append(dot)

    ax.legend()

    def update(t):
        for i, flight in enumerate(flights):
            # Select all points up to current time step t
            path = flight["path"][:t+1]
            if not path:
                continue
            xs, ys, zs, ts = zip(*path)
            flight_lines[i].set_data(xs, ys)
            flight_lines[i].set_3d_properties(zs)

            # Dot at current position
            flight_dots[i].set_data(xs[-1:], ys[-1:])
            flight_dots[i].set_3d_properties(zs[-1:])
        return flight_lines + flight_dots

    anim = FuncAnimation(fig, update, frames=T_MAX, interval=800, blit=True)

    os.makedirs('images', exist_ok=True)
    anim.save('images/flight_paths_3d.gif', writer='imagemagick')
    print("3D animation saved as images/flight_paths_3d.gif")

if __name__ == "__main__":
    grid = create_4d_grid()
    print(f"Total Nodes: {len(grid.nodes)}")
    print(f"Total Edges: {len(grid.edges)}")

    flights = define_flights()
    animate_flights_3d(flights)
