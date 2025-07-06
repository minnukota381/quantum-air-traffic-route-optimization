import time
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qiskit.primitives import BackendSampler
from qiskit_aer import Aer

from typing import Dict, Tuple, List

# === Problem Parameters ===
X_DIM, Y_DIM, Z_DIM, T_MAX = 2, 2, 1, 2

flights = [
    {"id": "F1", "start": (0, 0, 0, 0), "end": (1, 1, 0, 1)},
    {"id": "F2", "start": (1, 0, 0, 0), "end": (0, 1, 0, 1)},
    {"id": "F3", "start": (0, 1, 0, 0), "end": (1, 0, 0, 1)},
]

# === Step 1: Define Binary Variables ===
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
    return qp, variable_map

# === Step 2: Add Constraints ===
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

# === Step 3: Solve with QAOA and Classical Optimizer ===
def solve_qaoa(qubo):
    algorithm_globals.random_seed = 42
    sampler = BackendSampler(Aer.get_backend("qasm_simulator"))
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    start = time.time()
    result = optimizer.solve(qubo)
    elapsed = time.time() - start
    return result, elapsed

def solve_classically(qubo):
    try:
        from qiskit_optimization.algorithms import CplexOptimizer
        solver = CplexOptimizer()
    except Exception:
        print("⚠️ CPLEX not found. Falling back to NumPyMinimumEigensolver.")
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    start = time.time()
    result = solver.solve(qubo)
    elapsed = time.time() - start
    return result, elapsed

# === Step 4: Visualize ===
def extract_flight_paths(result) -> Dict[str, List[Tuple[int, int, int, int]]]:
    path_map = {}
    for var, val in result.variables_dict.items():
        if val == 1 and var.startswith("x_"):
            _, fid, x, y, z, t = var.split("_")
            path_map.setdefault(fid, []).append((int(x), int(y), int(z), int(t)))
    for path in path_map.values():
        path.sort(key=lambda pos: pos[3])
    return path_map

def plot_paths(qaoa_paths, classical_paths):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"F1": "blue", "F2": "green", "F3": "red"}

    for idx, (title, path_map) in enumerate(
        [("QAOA Paths", qaoa_paths), ("Classical Paths", classical_paths)]
    ):
        ax = axes[idx]
        for fid, steps in path_map.items():
            xs, ys = zip(*[(x, y) for x, y, z, t in steps])
            ax.plot(xs, ys, marker="o", label=fid, color=colors[fid])
        ax.set_xlim(-0.5, X_DIM - 0.5)
        ax.set_ylim(-0.5, Y_DIM - 0.5)
        ax.set_title(title)
        ax.set_xticks(range(X_DIM))
        ax.set_yticks(range(Y_DIM))
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# === Main Runner ===
if __name__ == "__main__":
    qp, var_map = define_binary_variables(flights)
    add_conflict_constraints(qp, var_map, flights)
    add_delay_penalties(qp, var_map, flights)
    add_path_continuity_constraints(qp, var_map, flights)
    print("✅ QUBO problem formulated.")

    qubo = QuadraticProgramToQubo().convert(qp)

    print("\n🚀 Solving with QAOA...")
    qaoa_result, t_qaoa = solve_qaoa(qubo)

    print("\n💻 Solving Classically...")
    classical_result, t_classical = solve_classically(qubo)

    print("\n📊 Results:")
    print(f"QAOA Cost: {qaoa_result.fval:.2f} | Time: {t_qaoa:.2f}s")
    print(f"Classical Cost: {classical_result.fval:.2f} | Time: {t_classical:.2f}s")

    print("\n✅ QAOA Solution:")
    for var, val in qaoa_result.variables_dict.items():
        if val == 1:
            print(var)

    print("\n✅ Classical Solution:")
    for var, val in classical_result.variables_dict.items():
        if val == 1:
            print(var)

    qaoa_paths = extract_flight_paths(qaoa_result)
    classical_paths = extract_flight_paths(classical_result)

    plot_paths(qaoa_paths, classical_paths)
