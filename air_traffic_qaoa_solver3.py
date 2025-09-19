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
# Config (3 flights, delays enabled)
# -------------------
X_DIM, Y_DIM, Z_DIM = 2, 2, 1
T_MAX = 3  # t=0 (start), t=1 (decision), t=2 (can arrive late)

# Penalties
P_ONEHOT     = 80.0   # exactly 1 position per (flight, t)
P_CONFLICT   = 60.0   # same-cell conflict avoidance
P_CONTINUITY = 15.0   # penalize non-neighbor transitions (forces adjacency)
P_DELAY      = 30.0   # penalty for arriving at t=2 instead of deadline t=1
P_MOVE       = 2.0    # small fuel proxy: moving costs more than waiting
P_ANCHOR     = 20.0   # tiny nudge toward start/end (already hard enforced)

SEP_RADIUS_CELLS = 0  # set to 1 to add extra spacing beyond same cell
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
# Variables: hard start at t=0, full grid at t=1, hard end at t=2
# -------------------
def define_binary_variables(qp, flights):
    var_map = {}
    onehot = {}
    for f in flights:
        fid = f["id"]
        sx, sy, sz, st = f["start"]  # st must be 0
        ex, ey, ez, et = f["end"]    # et = 1 (deadline), but we allow late at t=2

        for t in range(T_MAX):
            group = []
            if t == 0:
                name = f"x_{fid}_{sx}_{sy}_{sz}_{t}"
                qp.binary_var(name); var_map[(fid,sx,sy,sz,t)] = name; group.append(name)
            elif t == 1:
                for (x,y,z) in all_positions():
                    name = f"x_{fid}_{x}_{y}_{z}_{t}"
                    qp.binary_var(name); var_map[(fid,x,y,z,t)] = name; group.append(name)
            elif t == 2:
                # allow only end cell at t=2 (if late)
                name = f"x_{fid}_{ex}_{ey}_{ez}_{t}"
                qp.binary_var(name); var_map[(fid,ex,ey,ez,t)] = name; group.append(name)
            onehot[(fid,t)] = group
    return var_map, onehot

# -------------------
# Penalty encodings
# -------------------
def add_onehot(obj, onehot_groups, penalty=P_ONEHOT):
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
            vars_here = []
            for fid in fids:
                key = (fid,x,y,z,t)
                if key in var_map: vars_here.append(var_map[key])
            for i in range(len(vars_here)):
                for j in range(i+1,len(vars_here)):
                    add_q(obj, vars_here[i], vars_here[j], penalty)

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
    # Penalize NON-neighbor transitions (cont), and add small move cost for moving (fuel proxy)
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX-1):
            # available positions at t and t+1
            pos_t   = [(x,y,z) for (x,y,z) in all_positions() if (fid,x,y,z,t)   in var_map]
            pos_tp1 = [(x,y,z) for (x,y,z) in all_positions() if (fid,x,y,z,t+1) in var_map]
            for (x1,y1,z1) in pos_t:
                v1 = var_map[(fid,x1,y1,z1,t)]
                neigh = set(neighbors_including_wait(x1,y1,z1))
                for (x2,y2,z2) in pos_tp1:
                    v2 = var_map[(fid,x2,y2,z2,t+1)]
                    if (x2,y2,z2) in neigh:
                        # neighbor/wait: no continuity penalty; add small move fuel if moved
                        if (x2,y2,z2) != (x1,y1,z1):
                            add_q(obj, v1, v2, move_pen)
                    else:
                        # non-neighbor: penalize to discourage teleportation
                        add_q(obj, v1, v2, cont)

def add_delay(obj, var_map, flights, delay_penalty=P_DELAY):
    # If a flight reaches end at t=2, it's late vs deadline=1 → linear penalty on that end var
    for f in flights:
        fid = f["id"]
        ex,ey,ez,deadline = f["end"]  # we set deadline=1
        t_late = 2
        key = (fid, ex,ey,ez, t_late)
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

def plot_paths(result, flights, save_path="paths_3d.png"):
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
    ax.set_title("✈️ Flight Paths with Delay & Safety (X,Y vs Time)")
    ax.legend(); plt.tight_layout(); fig.savefig(save_path, dpi=130)
    print(f"Saved plot to {save_path}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    flights = [
        # Deadline is t=1 (on-time). Arriving at t=2 is allowed but penalized = delay.
        {"id": "F1", "start": (0,0,0,0), "end": (1,1,0,1)},
        {"id": "F2", "start": (1,0,0,0), "end": (0,1,0,1)},
        {"id": "F3", "start": (0,1,0,0), "end": (1,0,0,1)},
    ]

    qp = QuadraticProgram("ATC_QUBO_Delay_3Flights")
    var_map, onehot = define_binary_variables(qp, flights)

    total_qubits = len(var_map)
    print(f"Total binary variables (qubits): {total_qubits}")
    if total_qubits > MAX_QUBITS_REAL_HW:
        print(f"⚠️ {total_qubits} > {MAX_QUBITS_REAL_HW}. Use simulator or shrink.")

    obj = new_obj()
    add_onehot(obj, onehot, penalty=P_ONEHOT)
    add_start_end_nudges(obj, var_map, flights, anchor=P_ANCHOR)
    add_conflicts(obj, var_map, flights, penalty=P_CONFLICT)
    if SEP_RADIUS_CELLS > 0:
        add_separation_radius(obj, var_map, flights, radius=SEP_RADIUS_CELLS, penalty=P_CONFLICT)
    add_continuity_and_fuel(obj, var_map, flights, cont=P_CONTINUITY, move_pen=P_MOVE)
    add_delay(obj, var_map, flights, delay_penalty=P_DELAY)

    qp.minimize(constant=obj["const"], linear=obj["linear"], quadratic=obj["quadratic"])
    print("✅ QUBO with delay, safety, and fuel proxy formulated.")

    result = solve_qubo(qp)
    print("\n🧠 QAOA Solution (Aer Sampler):")
    print(result)
    print(f"\n🎯 Objective value (cost): {result.fval:.2f}")

    # Feasibility check: exactly one cell per (flight, t)
    for f in flights:
        fid = f["id"]
        for t in range(T_MAX):
            ones = 0
            for (x,y,z) in all_positions():
                var = f"x_{fid}_{x}_{y}_{z}_{t}"
                ones += int(result.variables_dict.get(var, 0))
            if ones != 1:
                print(f"⚠️ Feasibility warning: {fid} has {ones} active cells at t={t} (should be 1)")

    # Arrival time report (on-time if end@t=1, late if end@t=2)
    for f in flights:
        fid = f["id"]; ex,ey,ez,_ = f["end"]
        on_time = result.variables_dict.get(f"x_{fid}_{ex}_{ey}_{ez}_1", 0) == 1
        late    = result.variables_dict.get(f"x_{fid}_{ex}_{ey}_{ez}_2", 0) == 1
        status = "ON-TIME (t=1)" if on_time else ("LATE (t=2)" if late else "UNKNOWN")
        print(f"{fid} arrival status: {status}")

    plot_paths(result, flights)
