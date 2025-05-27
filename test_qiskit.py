from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a 2-qubit quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Obtain the statevector using the Statevector class
statevector = Statevector.from_instruction(qc)

print("Quantum statevector:", statevector)

