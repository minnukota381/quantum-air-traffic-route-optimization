import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_aer.primitives import Sampler as AerSampler
# If you ever hit a qubit coupling limit:
# from qiskit_aer import AerSimulator
# sampler = AerSampler(backend=AerSimulator())

# -------------------
# Config
# -------------------
X_DIM, Y_DIM, Z_DIM, T_MAX = 2, 2, 1, 2  # t=0..1 (short horizon)
P_ONEHOT     = 600.0     # strong: exactly 1 per (flight, t)
P_CONFLICT   = 300.0     # forbid co-location
P_SEPARATION = 0.0       # set >0 and SEP_RADIUS_CELLS>0 to keep distance
P_CONTINUITY = 12.0      # reward valid neighbor/wait moves
P_ANCHOR     = 60.0      # soft nudge (kept small now)
P_DELAY      = 25.0      # only matters if T_MAX allows lateness
SEP_RADIUS_CELLS = 0
MAX_QUBITS_REAL_HW = 27

# -------------------
# Helpers
# -------------------
def neighbors_including_wait(x, y, z):
    deltas = [-1, 0, 1]
    out = []
    for dx in deltas:
        for dy in deltas:
            for dz in deltas:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X_DIM and 0 <= ny < Y_DIM and 0 <= nz < Z_DIM:
                    out.append((nx, ny, nz))
    return out

def all_positions():
    for x in range(X_DIM):
        for y in range(Y_DIM):
            for z in range(Z_DIM):
                yield (x, y, z)

def new_objective_accumulator():
    return {"const": 0.0, "linear": {}, "quadratic": {}}

def add_const(obj, c): obj["const"] += c
def add_lin(obj, v, c): obj["linear"][v] = obj["linear"].get(v, 0.0) + c
def add_quad(obj, vi, vj, c):
    a, b = (vi, vj) if vi <= vj else (vj, vi)
    obj["quadratic"][(a, b)] = obj["quadratic"].get((a, b), 0.0) + c

# -------------------
# Variables
# -------------------
def define_binary_variables(qp, flights):
    """
    Critically: at the flight's start time, create ONLY the start cell var.
               at the flight's end time,   create ONLY the end   cell var.
    That hard-enforces start/end without huge penalties.
    """
    var_map = {}
    onehot_groups = {}
    for f in flights:
        fid = f["id"]
        sx, sy, sz, st = f["start"]
        ex, ey, ez, et = f["end"]
        for t in range(T_MAX):
            group = []
            if t == st:
                # only the true start cell is allowed
                name = f"x_{fid}_{sx}_{sy}_{sz}_{t}"
                qp.binary_var(name)
                var_map[(fid, sx, sy, sz, t)] = name
                group.append(name)
            elif t == et:
                # only the true end cell is allowed
                name = f"x_{fid}_{ex}_{ey}_{ez}_{t}"
                qp.binary_var(name)
                var_map[(fid, ex, ey, ez, t)] = name
                group.append(name)
            else:
                # normal: all cells are allowed
                for (x, y, z) in all_positions():
                    name = f"x_{fid}_{x}_{y}_{z}_{t}"
                    qp.binary_var(name)
                    var_map[(fid, x, y, z, t)] = name
                    group.append(name)
            onehot_groups[(fid, t)] = group
    return var_map, onehot_groups

# -------------------
# Penalties
# -------------------
def add_onehot_per_flight_time(obj, onehot_groups, penalty=P_ONEHOT):
    for _, vars_at_ft in onehot_groups.items():
        # (sum x_i - 1)^2 = sum x_i + 2*sum_{i<j} x_i x_j - 2*sum x_i + 1
        for v in vars_at_ft:
            add_lin(obj, v, -penalty)  # -penalty * x_i
        for i in range(len(vars_at_ft)):
            for j in range(i + 1, len(vars_at_ft)):
                add_quad(obj, vars_at_ft[i], vars_at_ft[j], 2.0 * penalty)
        add_const(obj, penalty)

def add_start_end_anchors(obj, var_map, flights, anchor=P_ANCHOR):
    # small nudge (start/end are already enforced structurally)
    for f in flights:
        fid = f["id"]
        sx, sy, sz, st = f["start"]
        ex, ey, ez, et = f["end"]
        add_lin(obj, var_map[(fid, sx, sy, sz, st)], -anchor)
        add_lin(obj, var_map[(fid, ex, ey, ez, et)], -anchor)

def add_same_cell_conflicts(obj, var_map, flights, penalty=P_CONFLICT):
    fids = [f["id"] for f in flights]
    for t in range(T_MAX):
        for (x, y, z) in all_positions():
            # if some flights don't have the variable at this time (because it’s start/end for them),
            # skip them safely using .get checks below.
            vars_here = []
            for fid in fids:
                key = (fid, x, y, z, t)
                if key in var_map:
                    vars_here.append(var_map[key])
            for i in range(len(vars_here)):
                for j in range(i + 1, len(vars_here)):
                    add_quad(obj, vars_here[i], vars_here[j], penalty)

def add_path_continuity(obj, var_map, flights, cont=P_CONTINUITY):
    # reward only valid neighbor (incl. wait) transitions t -> t+1
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX - 1):
            # list available cells at t and t+1 for this flight
            pos_t   = [(x, y, z) for (x, y, z) in all_positions() if (fid, x, y, z, t) in var_map]
            pos_tp1 = [(x, y, z) for (x, y, z) in all_positions() if (fid, x, y, z, t+1) in var_map]
            for (x1, y1, z1) in pos_t:
                v1 = var_map[(fid, x1, y1, z1, t)]
                for (nx, ny, nz) in neighbors_including_wait(x1, y1, z1):
                    if (fid, nx, ny, nz, t+1) in var_map:
                        v2 = var_map[(fid, nx, ny, nz, t+1)]
                        add_quad(obj, v1, v2, -cont)

# (Delay penalty kept for completeness; won’t trigger with T_MAX=2 & deadline=1)
def add_delay_penalties(obj, var_map, flights, delay_penalty=P_DELAY):
    for f in flights:
        fid = f["id"]
        ex, ey, ez, t_end_anchor = f["end"]
        deadline = f.get("deadline", t_end_anchor)
        for t in range(deadline + 1, T_MAX):
            key = (fid, ex, ey, ez, t)
            if key in var_map:
                add_lin(obj, var_map[key], delay_penalty)

# -------------------
# Solve & plot
# -------------------
def solve_qubo(qp):
    qubo = QuadraticProgramToQubo().convert(qp)
    algorithm_globals.random_seed = 42
    sampler = AerSampler()
    sampler.options.shots = 4096
    sampler.options.seed_simulator = 42
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=2)
    optimizer = MinimumEigenOptimizer(qaoa)
    return optimizer.solve(qubo)

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

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    flights = [
        {"id": "F1", "start": (0, 0, 0, 0), "end": (1, 1, 0, 1), "deadline": 1},
        {"id": "F2", "start": (1, 0, 0, 0), "end": (0, 1, 0, 1), "deadline": 1},
        {"id": "F3", "start": (0, 1, 0, 0), "end": (1, 0, 0, 1), "deadline": 1},
    ]

    qp = QuadraticProgram("AirTrafficQUBO_HardStartEnd")
    var_map, onehot_groups = define_binary_variables(qp, flights)

    total_qubits = len(var_map)
    print(f"Total binary variables (qubits): {total_qubits}")
    if total_qubits > MAX_QUBITS_REAL_HW:
        print(f"⚠️ {total_qubits} > {MAX_QUBITS_REAL_HW}. Use Aer simulator or shrink F/X/Y/Z/T_MAX.")

    obj = new_objective_accumulator()
    add_onehot_per_flight_time(obj, onehot_groups, penalty=P_ONEHOT)
    add_start_end_anchors(obj, var_map, flights, anchor=P_ANCHOR)
    add_same_cell_conflicts(obj, var_map, flights, penalty=P_CONFLICT)
    # add separation if you want extra buffer (>=1)
    if SEP_RADIUS_CELLS > 0:
        # simple adjacency separation not included here to keep it small; can be added if needed
        pass
    add_path_continuity(obj, var_map, flights, cont=P_CONTINUITY)
    add_delay_penalties(obj, var_map, flights, delay_penalty=P_DELAY)

    qp.minimize(constant=obj["const"], linear=obj["linear"], quadratic=obj["quadratic"])
    print("✅ QUBO with **structural** start/end + safety formulated.")

    result = solve_qubo(qp)

    print("\n🧠 QAOA Solution (Aer Sampler):")
    print(result)
    print(f"\n🎯 Objective value (cost): {result.fval:.2f}")

    # Feasibility: exactly 1 per (flight, t)
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX):
            ones = 0
            for (x, y, z) in all_positions():
                var = f"x_{fid}_{x}_{y}_{z}_{t}"
                ones += int(result.variables_dict.get(var, 0))
            if ones != 1:
                print(f"⚠️ Feasibility warning: {fid} has {ones} active cells at t={t} (should be 1)")

    print("\n🚀 Selected variables (1-bits):")
    for var, val in result.variables_dict.items():
        if val == 1:
            print(var)

    plot_3d_paths(result, flights)
