import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# === Airspace Dimensions ===
X_DIM = 5  # east-west
Y_DIM = 5  # north-south
Z_DIM = 3  # altitude
T_MAX = 10  # time steps

# === Step 1: Create 4D Grid ===
def create_4d_grid():
    G = nx.DiGraph()
    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    node = (x, y, z, t)
                    G.add_node(node)
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,1), (0,0,-1), (0,0,0)]:
                        nx_ = x + dx
                        ny_ = y + dy
                        nz_ = z + dz
                        if 0 <= nx_ < X_DIM and 0 <= ny_ < Y_DIM and 0 <= nz_ < Z_DIM:
                            next_node = (nx_, ny_, nz_, t+1)
                            if t+1 < T_MAX:
                                G.add_edge(node, next_node)
    return G

# === Step 2: Define Flights ===
def define_flights():
    return [
        {"id": "F1", "start": (0, 0, 0, 0), "end": (4, 4, 0, 9)},
        {"id": "F2", "start": (4, 0, 1, 0), "end": (0, 4, 1, 9)},
    ]

# === Step 3: Find Paths for Flights ===
def find_paths(graph, flights):
    paths = {}
    for flight in flights:
        try:
            path = nx.shortest_path(graph, source=flight['start'], target=flight['end'])
            paths[flight['id']] = path
            print(f"Flight {flight['id']} path found with {len(path)} steps.")
        except nx.NetworkXNoPath:
            print(f"Flight {flight['id']} has no available path.")
            paths[flight['id']] = []
    return paths

# === Step 4: 3D Animation ===
def animate_3d_paths(flight_paths, save_path='./images/flight_paths_3d.gif'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'b', 'g', 'm', 'c']

    max_time = T_MAX
    flight_ids = list(flight_paths.keys())

    def update(frame):
        ax.clear()
        for i, flight_id in enumerate(flight_ids):
            path = flight_paths[flight_id]
            if not path:
                continue
            xs, ys, zs = [], [], []
            for node in path:
                if node[3] <= frame:
                    xs.append(node[0])
                    ys.append(node[1])
                    zs.append(node[2])
            ax.plot(xs, ys, zs, color=colors[i % len(colors)], label=flight_id)
        ax.set_xlim(0, X_DIM - 1)
        ax.set_ylim(0, Y_DIM - 1)
        ax.set_zlim(0, Z_DIM - 1)
        ax.set_title(f"Time Step: {frame}")
        ax.set_xlabel("X (Longitude)")
        ax.set_ylabel("Y (Latitude)")
        ax.set_zlabel("Z (Altitude)")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=max_time, interval=1000)
    os.makedirs('./images', exist_ok=True)
    ani.save(save_path, writer='pillow')
    print(f"3D animation saved as {save_path}")

# === Step 6: Collision Detection ===
def detect_collisions(flight_paths):
    collision_log = []
    time_position_map = {}  # { (x,y,z,t) : [flight_id, ...] }

    for flight_id, path in flight_paths.items():
        if not path:
            continue
        for node in path:
            key = node  # (x, y, z, t)
            if key not in time_position_map:
                time_position_map[key] = []
            time_position_map[key].append(flight_id)

    for position_time, flights_here in time_position_map.items():
        if len(flights_here) > 1:
            collision_log.append({
                'position': position_time[:3],
                'time': position_time[3],
                'flights': flights_here
            })

    return collision_log

# === Main ===
if __name__ == "__main__":
    grid = create_4d_grid()
    print(f"Total Nodes: {len(grid.nodes)}")
    print(f"Total Edges: {len(grid.edges)}")

    flights = define_flights()
    flight_paths = find_paths(grid, flights)

    animate_3d_paths(flight_paths)

    collisions = detect_collisions(flight_paths)
    if collisions:
        print("\n⚠️ Collisions Detected:")
        for c in collisions:
            print(f"  At position {c['position']} at time t={c['time']}: Flights {', '.join(c['flights'])}")
    else:
        print("\n✅ No collisions detected.")
