import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit.primitives import BackendSampler
from qiskit_aer import Aer

# === Grid + Time Settings ===
X_DIM, Y_DIM, Z_DIM, T_MAX = 2, 2, 1, 2  # 2x2x1 grid, 2 time steps

# === Define Binary Variables ===
def define_binary_variables(flights):
    qp = QuadraticProgram("AirTrafficQUBO")
    variable_map = {}
    for flight in flights:
        fid = flight["id"]
        for t in range(T_MAX):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    for z in range(Z_DIM):
                        var = f"x_{fid}_{x}_{y}_{z}_{t}"
                        qp.binary_var(name=var)
                        variable_map[(fid, x, y, z, t)] = var
    print(f"✅ Total binary variables defined: {len(variable_map)}")
    return qp, variable_map

# === Constraint Functions ===
def add_conflict_constraints(qp, variable_map, flights, penalty=10):
    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    vars_here = []
                    for f in flights:
                        key = (f["id"], x, y, z, t)
                        if key in variable_map:
                            vars_here.append(variable_map[key])
                    for i in range(len(vars_here)):
                        for j in range(i + 1, len(vars_here)):
                            qp.minimize(quadratic={(vars_here[i], vars_here[j]): penalty})

def add_delay_penalties(qp, variable_map, flights, delay_penalty=5):
    for f in flights:
        fid = f["id"]
        ex, ey, ez, t_target = f["end"]
        for t in range(t_target + 1, T_MAX):
            key = (fid, ex, ey, ez, t)
            if key in variable_map:
                qp.minimize(linear={variable_map[key]: delay_penalty})

def neighbors(x, y, z):
    deltas = [-1, 0, 1]
    return [(x + dx, y + dy, z + dz) for dx in deltas for dy in deltas for dz in deltas
            if not (dx == dy == dz == 0)
            and 0 <= x + dx < X_DIM and 0 <= y + dy < Y_DIM and 0 <= z + dz < Z_DIM]

def add_path_continuity_constraints(qp, variable_map, flights, penalty=15):
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX - 1):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    for z in range(Z_DIM):
                        current_key = (fid, x, y, z, t)
                        if current_key not in variable_map:
                            continue
                        v_now = variable_map[current_key]
                        next_valid = []
                        for nx, ny, nz in neighbors(x, y, z):
                            next_key = (fid, nx, ny, nz, t + 1)
                            if next_key in variable_map:
                                next_valid.append(variable_map[next_key])
                        if not next_valid:
                            qp.minimize(linear={v_now: penalty})
                            continue
                        for v_next in next_valid:
                            qp.minimize(quadratic={(v_now, v_next): -2 * penalty})
                            qp.minimize(linear={v_next: penalty})
                        qp.minimize(linear={v_now: penalty})

# === 3D Plotting ===
def plot_3d_paths(result, flights):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for i, flight in enumerate(flights):
        fid = flight["id"]
        color = colors[i % len(colors)]
        xs, ys, ts = [], [], []

        for t in range(T_MAX):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    for z in range(Z_DIM):
                        var = f"x_{fid}_{x}_{y}_{z}_{t}"
                        if result.variables_dict.get(var, 0) == 1:
                            xs.append(x)
                            ys.append(y)
                            ts.append(t)
                            ax.text(x, y, t, f"{fid}-t{t}", color=color, fontsize=9)

        ax.plot(xs, ys, ts, color=color, marker='o', label=f"Flight {fid}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")
    ax.set_title("✈️ Flight Paths in 3D (X, Y, Time)")
    ax.legend()
    plt.show()

# === Main ===
if __name__ == "__main__":
    flights = [
        {"id": "F1", "start": (0, 0, 0, 0), "end": (1, 1, 0, 1)},
        {"id": "F2", "start": (1, 0, 0, 0), "end": (0, 1, 0, 1)},
        {"id": "F3", "start": (0, 1, 0, 0), "end": (1, 0, 0, 1)},
    ]


    qp, var_map = define_binary_variables(flights)
    add_conflict_constraints(qp, var_map, flights)
    add_delay_penalties(qp, var_map, flights)
    add_path_continuity_constraints(qp, var_map, flights)
    print("✅ QUBO problem formulated.")

    qubo = QuadraticProgramToQubo().convert(qp)

    algorithm_globals.random_seed = 42
    backend = Aer.get_backend("qasm_simulator")
    sampler = BackendSampler(backend)

    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qubo)

    print("\n🧠 QAOA Solution (qasm_simulator):")
    print(result)
    print(f"\n🎯 Objective value (cost): {result.fval:.2f}")

    print("\n🚀 Selected Flight Path Variables:")
    for var, val in result.variables_dict.items():
        if val == 1:
            print(var)

    plot_3d_paths(result, flights)
