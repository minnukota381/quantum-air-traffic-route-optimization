import numpy as np
import matplotlib
matplotlib.use("Agg")  # comment out if you want interactive plots
import matplotlib.pyplot as plt

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_aer.primitives import Sampler as AerSampler

# =========================
# Config
# =========================
X_DIM, Y_DIM, Z_DIM, T_MAX = 2, 2, 1, 2  # 2x2x1 grid, times t=0..1
MAX_QUBITS_REAL_HW = 27  # IBM real hardware cap

P_ANCHOR = 100.0
P_ONEHOT = 50.0
P_CONFLICT = 30.0
P_CONTINUITY = 10.0

# =========================
# Helpers
# =========================
def neighbors(x, y, z):
    deltas = [-1, 0, 1]
    result = []
    for dx in deltas:
        for dy in deltas:
            for dz in deltas:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X_DIM and 0 <= ny < Y_DIM and 0 <= nz < Z_DIM:
                    result.append((nx, ny, nz))
    return result

def all_positions():
    for x in range(X_DIM):
        for y in range(Y_DIM):
            for z in range(Z_DIM):
                yield (x, y, z)

# =========================
# Variable creation
# =========================
def define_binary_variables(qp, flights):
    var_map = {}
    onehot_groups = {}
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX):
            grp = []
            for (x, y, z) in all_positions():
                name = f"x_{fid}_{x}_{y}_{z}_{t}"
                qp.binary_var(name)
                var_map[(fid, x, y, z, t)] = name
                grp.append(name)
            onehot_groups[(fid, t)] = grp
    return var_map, onehot_groups

# =========================
# Objective helpers
# =========================
def new_objective_accumulator():
    return {"const": 0.0, "linear": {}, "quadratic": {}}

def add_const(obj, c):
    obj["const"] += c

def add_lin(obj, v, c):
    obj["linear"][v] = obj["linear"].get(v, 0.0) + c

def add_quad(obj, vi, vj, c):
    a, b = (vi, vj) if vi <= vj else (vj, vi)
    key = (a, b)
    obj["quadratic"][key] = obj["quadratic"].get(key, 0.0) + c

# =========================
# Penalty encodings
# =========================
def add_onehot_per_flight_time(obj, onehot_groups, penalty=P_ONEHOT):
    for (_, _), vars_at_ft in onehot_groups.items():
        for v in vars_at_ft:
            add_lin(obj, v, -penalty)
        for i in range(len(vars_at_ft)):
            for j in range(i + 1, len(vars_at_ft)):
                add_quad(obj, vars_at_ft[i], vars_at_ft[j], 2.0 * penalty)
        add_const(obj, penalty)

def add_start_end_anchors(obj, var_map, flights, anchor=P_ANCHOR):
    for f in flights:
        fid = f["id"]
        sx, sy, sz, st = f["start"]
        ex, ey, ez, et = f["end"]
        add_lin(obj, var_map[(fid, sx, sy, sz, st)], -anchor)
        add_lin(obj, var_map[(fid, ex, ey, ez, et)], -anchor)

def add_same_cell_conflict_penalties(obj, var_map, flights, conflict=P_CONFLICT):
    fids = [f["id"] for f in flights]
    for t in range(T_MAX):
        for (x, y, z) in all_positions():
            vars_here = [var_map[(fid, x, y, z, t)] for fid in fids]
            for i in range(len(vars_here)):
                for j in range(i + 1, len(vars_here)):
                    add_quad(obj, vars_here[i], vars_here[j], conflict)

def add_path_continuity(obj, var_map, flights, cont=P_CONTINUITY):
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX - 1):
            pos_t = [(x, y, z) for (x, y, z) in all_positions()]
            pos_tp1 = [(x, y, z) for (x, y, z) in all_positions()]
            for (x1, y1, z1) in pos_t:
                v1 = var_map[(fid, x1, y1, z1, t)]
                for (x2, y2, z2) in pos_tp1:
                    v2 = var_map[(fid, x2, y2, z2, t + 1)]
                    add_quad(obj, v1, v2, cont)
            for (x1, y1, z1) in pos_t:
                v1 = var_map[(fid, x1, y1, z1, t)]
                for (nx, ny, nz) in neighbors(x1, y1, z1):
                    v2 = var_map[(fid, nx, ny, nz, t + 1)]
                    add_quad(obj, v1, v2, -cont)

# =========================
# Solve & visualize
# =========================
def solve_qubo(qp):
    qubo = QuadraticProgramToQubo().convert(qp)
    algorithm_globals.random_seed = 42

    from qiskit_aer.primitives import Sampler as AerSampler
    sampler = AerSampler()                 # ← no options= here
    # Set options on the object:
    sampler.options.shots = 4096
    sampler.options.seed_simulator = 42

    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=150), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qubo)
    return result

def plot_3d_paths(result, flights, save_path="paths_3d.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["red", "blue", "green", "purple", "orange"]
    for i, f in enumerate(flights):
        fid = f["id"]
        color = colors[i % len(colors)]
        xs, ys, ts = [], [], []
        for t in range(T_MAX):
            for (x, y, z) in all_positions():
                var = f"x_{fid}_{x}_{y}_{z}_{t}"
                if result.variables_dict.get(var, 0) == 1:
                    xs.append(x); ys.append(y); ts.append(t)
                    ax.text(x, y, t, f"{fid}@t{t}", color=color, fontsize=9)
        ax.plot(xs, ys, ts, color=color, marker="o", label=f"Flight {fid}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Time")
    ax.set_title("✈️ Flight Paths (X, Y, Time)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"Saved plot to {save_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    flights = [
        {"id": "F1", "start": (0, 0, 0, 0), "end": (1, 1, 0, 1)},
        {"id": "F2", "start": (1, 0, 0, 0), "end": (0, 1, 0, 1)},
        {"id": "F3", "start": (0, 1, 0, 0), "end": (1, 0, 0, 1)},
    ]

    qp = QuadraticProgram("AirTrafficQUBO")
    var_map, onehot_groups = define_binary_variables(qp, flights)

    total_qubits = len(var_map)
    print(f"Total binary variables (qubits): {total_qubits}")
    if total_qubits > MAX_QUBITS_REAL_HW:
        print(f"⚠️ {total_qubits} > {MAX_QUBITS_REAL_HW}. Use simulator or shrink problem.")

    obj = new_objective_accumulator()
    add_onehot_per_flight_time(obj, onehot_groups, penalty=P_ONEHOT)
    add_start_end_anchors(obj, var_map, flights, anchor=P_ANCHOR)
    add_same_cell_conflict_penalties(obj, var_map, flights, conflict=P_CONFLICT)
    add_path_continuity(obj, var_map, flights, cont=P_CONTINUITY)

    qp.minimize(constant=obj["const"], linear=obj["linear"], quadratic=obj["quadratic"])
    print("✅ QUBO formulated.")

    result = solve_qubo(qp)

    print("\n🧠 QAOA Solution (Sampler backend):")
    print(result)
    print(f"\n🎯 Objective value (cost): {result.fval:.2f}")

    print("\n🚀 Selected variables (1-bits):")
    for var, val in result.variables_dict.items():
        if val == 1:
            print(var)

    plot_3d_paths(result, flights)
