from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import L_BFGS_B

from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.mappers.second_quantization import LogarithmicMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.circuit.library import TwoLocal, RealAmplitudes

old_so = SpinOp(("XYZ", 1), spin=1/2)
old_converter = QubitConverter(LogarithmicMapper())

old_qubit_op = old_converter.convert(old_so)
print("qubit operator: ", old_qubit_op)

ansatz = RealAmplitudes(4)
print("ansatz: ", ansatz)

vqe = VQE(Sampler(), ansatz=ansatz, optimizer=L_BFGS_B())
old_eig = vqe.compute_minimum_eigenvalue(old_qubit_op)
print("eigenvalue:", old_eig)

print("--------")

from qiskit_nature.second_q.operators import SpinOp as NewSpinOp
from qiskit_nature.second_q.mappers import LogarithmicMapper as NewLM
from qiskit_nature.second_q.mappers import QubitConverter as NewQC

new_so = NewSpinOp({"X_0 Y_1 Z_2": 1}, spin=1/2, num_orbitals=4)
new_converter = NewQC(NewLM())

new_qubit_op = new_converter.convert(new_so)
print("qubit operator: ", new_qubit_op)

ansatz = RealAmplitudes(4)
print("ansatz: ", ansatz)

vqe = VQE(Sampler(), ansatz=ansatz, optimizer=L_BFGS_B())
new_eig = vqe.compute_minimum_eigenvalue(new_qubit_op)
print("eigenvalue:", new_eig)




