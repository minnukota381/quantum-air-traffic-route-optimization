# air_traffic_qaoa_solver4.py
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

# -------------------
# Config (4 flights, delays enabled; 24 qubits total)
# -------------------
X_DIM, Y_DIM, Z_DIM = 2, 2, 1
T_MAX = 3  # t=0 start (hard), t=1 decision layer, t=2 optional late-arrival end

# Penalties
P_ONEHOT     = 80.0     # exactly 1 position per (flight, t) — but ONLY for t=0,1
P_CONFLICT   = 60.0     # same-cell conflicts across flights at same time
P_CONTINUITY = 15.0     # penalize non-neighbor transitions (discourage teleport)
P_DELAY      = 30.0     # arriving at t=2 instead of t=1 (deadline) = penalty
P_MOVE       = 2.0      # tiny "fuel proxy": moving costs a little more than waiting
P_ANCHOR     = 20.0     # small nudge toward start and on-time end

SEP_RADIUS_CELLS = 0    # set to 1 if you want a spacing buffer beyond same-cell
MAX_QUBITS_REAL_HW = 27

# -------------------
# Helpers
# -------------------
def all_positions():
    for x in range(X_DIM):
        for y in range(Y_DIM):
            for z in range(Z_DIM):
                yield (x, y, z)

def neighbors_including_wait(x, y, z):
    deltas = [-1, 0, 1]
    out = []
    for dx in deltas:
        for dy in deltas:
            for dz in deltas:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X_DIM and 0 <= ny < Y_DIM and 0 <= nz < Z_DIM:
                    out.append((nx, ny, nz))
    return out  # includes (x,y,z) itself (wait)

def new_obj():
    return {"const": 0.0, "linear": {}, "quadratic": {}}
def add_c(o,c): o["const"] += c
def add_l(o,v,c): o["linear"][v] = o["linear"].get(v,0.0) + c
def add_q(o,vi,vj,c):
    a,b = (vi,vj) if vi<=vj else (vj,vi)
    o["quadratic"][(a,b)] = o["quadratic"].get((a,b),0.0) + c

# -------------------
# Variables: t=0 fixed start, t=1 full grid, t=2 only end (OPTIONAL)
# -------------------
def define_binary_variables(qp, flights):
    var_map = {}
    onehot_t01 = {}  # one-hot groups only for t=0 and t=1
    for f in flights:
        fid = f["id"]
        sx, sy, sz, _ = f["start"]  # t=0
        ex, ey, ez, _ = f["end"]    # deadline t=1; t=2 allowed (late)

        # t=0: only the true start cell (one variable)
        t = 0
        g0 = []
        name = f"x_{fid}_{sx}_{sy}_{sz}_{t}"
        qp.binary_var(name)
        var_map[(fid,sx,sy,sz,t)] = name
        g0.append(name)
        onehot_t01[(fid,t)] = g0

        # t=1: all grid cells (four variables)
        t = 1
        g1 = []
        for (x,y,z) in all_positions():
            name = f"x_{fid}_{x}_{y}_{z}_{t}"
            qp.binary_var(name)
            var_map[(fid,x,y,z,t)] = name
            g1.append(name)
        onehot_t01[(fid,t)] = g1

        # t=2: only the true end cell (OPTIONAL, so NOT in one-hot)
        t = 2
        name = f"x_{fid}_{ex}_{ey}_{ez}_{t}"
        qp.binary_var(name)
        var_map[(fid,ex,ey,ez,t)] = name

    return var_map, onehot_t01

# -------------------
# Penalty encodings
# -------------------
def add_onehot_t0_t1(obj, onehot_groups, penalty=P_ONEHOT):
    # exactly-one for t=0 and t=1 only
    for _, group in onehot_groups.items():
        for v in group: add_l(obj, v, -penalty)
        for i in range(len(group)):
            for j in range(i+1,len(group)):
                add_q(obj, group[i], group[j], 2.0*penalty)
        add_c(obj, penalty)

def add_start_end_nudges(obj, var_map, flights, anchor=P_ANCHOR):
    for f in flights:
        fid = f["id"]
        sx,sy,sz,_ = f["start"]
        ex,ey,ez,_ = f["end"]
        add_l(obj, var_map[(fid,sx,sy,sz,0)], -anchor)
        add_l(obj, var_map[(fid,ex,ey,ez,1)], -anchor)  # on-time preferred

def add_conflicts(obj, var_map, flights, penalty=P_CONFLICT):
    fids = [f["id"] for f in flights]
    for t in range(T_MAX):
        for (x,y,z) in all_positions():
            vs = []
            for fid in fids:
                key = (fid,x,y,z,t)
                if key in var_map: vs.append(var_map[key])
            for i in range(len(vs)):
                for j in range(i+1,len(vs)):
                    add_q(obj, vs[i], vs[j], penalty)

def add_separation_radius(obj, var_map, flights, radius=SEP_RADIUS_CELLS, penalty=P_CONFLICT):
    if radius <= 0: return
    fids = [f["id"] for f in flights]
    def manhattan(a,b): return sum(abs(a[i]-b[i]) for i in range(3))
    for t in range(T_MAX):
        positions = list(all_positions())
        for i,p in enumerate(positions):
            for j in range(i+1, len(positions)):
                q = positions[j]
                if manhattan(p,q) <= radius:
                    for a in range(len(fids)):
                        for b in range(a+1, len(fids)):
                            ka = (fids[a], p[0], p[1], p[2], t)
                            kb = (fids[b], q[0], q[1], q[2], t)
                            if ka in var_map and kb in var_map:
                                add_q(obj, var_map[ka], var_map[kb], penalty)

def add_continuity_and_fuel(obj, var_map, flights, cont=P_CONTINUITY, move_pen=P_MOVE):
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX-1):
            pos_t   = [(x,y,z) for (x,y,z) in all_positions() if (fid,x,y,z,t)   in var_map]
            pos_tp1 = [(x,y,z) for (x,y,z) in all_positions() if (fid,x,y,z,t+1) in var_map]
            for (x1,y1,z1) in pos_t:
                v1 = var_map[(fid,x1,y1,z1,t)]
                neigh = set(neighbors_including_wait(x1,y1,z1))
                for (x2,y2,z2) in pos_tp1:
                    v2 = var_map[(fid,x2,y2,z2,t+1)]
                    if (x2,y2,z2) in neigh:
                        if (x2,y2,z2) != (x1,y1,z1):  # moved (fuel proxy)
                            add_q(obj, v1, v2, move_pen)
                    else:
                        add_q(obj, v1, v2, cont)

def add_delay_linear(obj, var_map, flights, delay_penalty=P_DELAY):
    # If end at t=2 chosen → add linear penalty
    for f in flights:
        fid = f["id"]
        ex,ey,ez,_ = f["end"]  # deadline is t=1
        key = (fid, ex,ey,ez, 2)
        if key in var_map:
            add_l(obj, var_map[key], delay_penalty)

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

def plot_paths(result, flights, save_path="paths_4flights_3t.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["red","blue","green","purple","orange"]
    for i,f in enumerate(flights):
        fid = f["id"]; color = colors[i % len(colors)]
        xs, ys, ts = [], [], []
        for t in range(T_MAX):
            for (x,y,z) in all_positions():
                var = f"x_{fid}_{x}_{y}_{z}_{t}"
                if result.variables_dict.get(var, 0) == 1:
                    xs.append(x); ys.append(y); ts.append(t)
                    ax.text(x, y, t, f"{fid}@t{t}", color=color, fontsize=9)
        ax.plot(xs, ys, ts, color=color, marker="o", label=f"Flight {fid}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Time")
    ax.set_title("✈️ 4 Flights with Delay & Safety (X,Y vs Time)")
    ax.legend(); plt.tight_layout(); fig.savefig(save_path, dpi=130)
    print(f"Saved plot to {save_path}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    flights = [
        # All deadlines are t=1; arriving at t=2 is allowed but penalized (delay)
        {"id": "F1", "start": (0,0,0,0), "end": (1,1,0,1)},
        {"id": "F2", "start": (1,0,0,0), "end": (0,1,0,1)},
        {"id": "F3", "start": (0,1,0,0), "end": (1,0,0,1)},
        {"id": "F4", "start": (1,1,0,0), "end": (0,0,0,1)},
    ]

    qp = QuadraticProgram("ATC_QUBO_Delay_4Flights")
    var_map, onehot_t01 = define_binary_variables(qp, flights)

    total_qubits = len(var_map)  # should be 4 flights * (1 + 4 + 1) = 24
    print(f"Total binary variables (qubits): {total_qubits}")
    if total_qubits > MAX_QUBITS_REAL_HW:
        print(f"⚠️ {total_qubits} > {MAX_QUBITS_REAL_HW}. Use simulator or shrink.")

    obj = new_obj()
    add_onehot_t0_t1(obj, onehot_t01, penalty=P_ONEHOT)                       # exactly-one at t=0,1
    add_start_end_nudges(obj, var_map, flights, anchor=P_ANCHOR)               # prefer on-time end
    add_conflicts(obj, var_map, flights, penalty=P_CONFLICT)                   # same-cell safety
    if SEP_RADIUS_CELLS > 0:
        add_separation_radius(obj, var_map, flights, SEP_RADIUS_CELLS, P_CONFLICT)
    add_continuity_and_fuel(obj, var_map, flights, P_CONTINUITY, P_MOVE)       # path realism + fuel
    add_delay_linear(obj, var_map, flights, P_DELAY)                           # penalize late arrivals

    qp.minimize(constant=obj["const"], linear=obj["linear"], quadratic=obj["quadratic"])
    print("✅ QUBO with delay, safety, and fuel proxy (4 flights) formulated.")

    result = solve_qubo(qp)
    print("\n🧠 QAOA Solution (Aer Sampler):")
    print(result)
    print(f"\n🎯 Objective value (cost): {result.fval:.2f}")

    # Feasibility checks:
    for f in flights:
        fid = f["id"]
        # t=0 and t=1 must have exactly one:
        for t in [0,1]:
            ones = 0
            for (x,y,z) in all_positions():
                var = f"x_{fid}_{x}_{y}_{z}_{t}"
                ones += int(result.variables_dict.get(var, 0))
            if ones != 1:
                print(f"⚠️ Feasibility warning: {fid} has {ones} active cells at t={t} (should be 1)")
        # t=2 is OPTIONAL (at-most-one); report status:
        ex,ey,ez,_ = f["end"]
        late = int(result.variables_dict.get(f"x_{fid}_{ex}_{ey}_{ez}_2", 0))
        if late > 1:
            print(f"⚠️ Feasibility warning: {fid} has >1 at t=2 (should be ≤1)")
        on_time = result.variables_dict.get(f"x_{fid}_{ex}_{ey}_{ez}_1", 0) == 1
        status = "ON-TIME (t=1)" if on_time else ("LATE (t=2)" if late == 1 else "UNKNOWN")
        print(f"{fid} arrival status: {status}")

    plot_paths(result, flights)
