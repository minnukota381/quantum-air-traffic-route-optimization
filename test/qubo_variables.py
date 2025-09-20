from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

# Airspace dimensions
X_DIM, Y_DIM, Z_DIM, T_MAX = 5, 5, 3, 10

# === STEP 1: Define Binary Variables ===
def define_binary_variables(flights):
    qp = QuadraticProgram(name="AirTrafficQUBO")
    variable_map = {}

    for flight in flights:
        fid = flight['id']
        for t in range(T_MAX):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    for z in range(Z_DIM):
                        var_name = f"x_{fid}_{x}_{y}_{z}_{t}"
                        qp.binary_var(name=var_name)
                        variable_map[(fid, x, y, z, t)] = var_name

    print(f"✅ Total binary variables defined: {len(variable_map)}")
    return qp, variable_map

# === STEP 2: Conflict Avoidance Constraint ===
def add_conflict_constraints(qp, variable_map, flights, penalty=10):
    for t in range(T_MAX):
        for x in range(X_DIM):
            for y in range(Y_DIM):
                for z in range(Z_DIM):
                    vars_at_position = []
                    for flight in flights:
                        key = (flight['id'], x, y, z, t)
                        if key in variable_map:
                            vars_at_position.append(variable_map[key])
                    for i in range(len(vars_at_position)):
                        for j in range(i + 1, len(vars_at_position)):
                            var1 = vars_at_position[i]
                            var2 = vars_at_position[j]
                            qp.minimize(linear={}, quadratic={(var1, var2): penalty})

# === STEP 3: Delay Penalties ===
def add_delay_penalties(qp, variable_map, flights, delay_penalty=5):
    for flight in flights:
        fid = flight['id']
        ex, ey, ez, expected_t = flight['end']
        for t in range(expected_t + 1, T_MAX):
            key = (fid, ex, ey, ez, t)
            if key in variable_map:
                var = variable_map[key]
                qp.minimize(linear={var: delay_penalty})

# === STEP 4: Path Continuity Constraints ===
def neighbors(x, y, z):
    deltas = [-1, 0, 1]
    return [(x + dx, y + dy, z + dz)
            for dx in deltas for dy in deltas for dz in deltas
            if not (dx == dy == dz == 0)
            and 0 <= x + dx < X_DIM
            and 0 <= y + dy < Y_DIM
            and 0 <= z + dz < Z_DIM]

def add_path_continuity_constraints(qp, variable_map, flights, penalty=15):
    for flight in flights:
        fid = flight['id']
        for t in range(T_MAX - 1):
            for x in range(X_DIM):
                for y in range(Y_DIM):
                    for z in range(Z_DIM):
                        current_key = (fid, x, y, z, t)
                        if current_key not in variable_map:
                            continue
                        current_var = variable_map[current_key]

                        valid_next_vars = []
                        for nx, ny, nz in neighbors(x, y, z):
                            next_key = (fid, nx, ny, nz, t + 1)
                            if next_key in variable_map:
                                valid_next_vars.append(variable_map[next_key])

                        if not valid_next_vars:
                            qp.minimize(linear={current_var: penalty})
                            continue

                        for next_var in valid_next_vars:
                            qp.minimize(linear={}, quadratic={(current_var, next_var): -2 * penalty})
                            qp.minimize(linear={next_var: penalty})
                        qp.minimize(linear={current_var: penalty})

# === STEP 5: Build and Solve QUBO with QAOA ===
if __name__ == "__main__":
    # Define flights
    flights = [
        {"id": "F1", "start": (0, 0, 0, 0), "end": (4, 4, 0, 9)},
        {"id": "F2", "start": (4, 0, 1, 0), "end": (0, 4, 1, 9)},
    ]

    # Build QUBO problem
    qp, var_map = define_binary_variables(flights)
    add_conflict_constraints(qp, var_map, flights)
    add_delay_penalties(qp, var_map, flights)
    add_path_continuity_constraints(qp, var_map, flights)

    print("✅ QUBO problem formulated.")

    # Convert to QUBO format for QAOA
    qubo_converter = QuadraticProgramToQubo()
    qubo_problem = qubo_converter.convert(qp)

    # Setup QAOA solver using Sampler
    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)

    # Solve the QUBO
    result = optimizer.solve(qubo_problem)

    # Display solution
    print("\n🧠 QAOA Solution found:")
    print(result)

    # Extract active variables
    print("\n🚀 Flight Path Variables Selected by QAOA (value = 1):")
    for var, val in result.variables_dict.items():
        if val == 1:
            print(var)
