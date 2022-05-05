
import numpy as np
from qiskit.chemistry.drivers import HDF5Driver as AquaHDF5Driver
from qiskit.chemistry.transformations import FermionicTransformation
from qiskit.chemistry.transformations.fermionic_transformation import \
    FermionicQubitMappingType
from qiskit.chemistry.algorithms.ground_state_solvers import OrbitalOptimizationVQE as OOVQE_Aqua
from qiskit.chemistry.algorithms.ground_state_solvers.orbital_optimization_vqe import OrbitalRotation as OR_Aqua

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.oovqe_algorithm import (
    OrbitalOptimizationVQE,
    CustomProblem,
)

from qiskit_nature.algorithms.ground_state_solvers.custom_problem import CustomProblem

############################
# test orbital rotations aqua

parameters_1 = np.asarray([ 0.79045546, -0.04134567,  0.04944946, -0.02971617, -0.00374005, 0.77542149])

matrix_a_1 = np.asarray([[ 0.70185237,  0.70941354, -0.06073103,  0.02115215],
 [-0.71035342,  0.70331919, -0.0031398,  -0.02702369],
 [ 0.00504813,  0.03518248,  0.71315562,  0.70010411],
 [-0.05268571, -0.02892638, -0.69836304,  0.71321563]])

matrix_b_1 = np.asarray([[ 0.70185237,  0.70941354, -0.06073103,  0.02115215],
 [-0.71035342,  0.70331919, -0.0031398,  -0.02702369],
 [ 0.00504813,  0.03518248,  0.71315562,  0.70010411],
 [-0.05268571, -0.02892638, -0.69836304,  0.71321563]])

############################
# test_orbital_rotations nature

parameters_2 =  [ 0.039374,   -0.47225463, -0.61891996,  0.02598386,  0.79045546, -0.04134567,
  0.04944946, -0.02971617, -0.00374005,  0.77542149]

matrix_a_2 = [[ 0.72123296,  0.2553158,  -0.43551614, -0.47430218],
 [ 0.19648728,  0.71249029 , 0.00955327,  0.67354218],
 [ 0.41338402,  0.0366419,   0.89340815, -0.17202586],
 [ 0.51993214 ,-0.65255559, -0.10980116,  0.54017172]]

matrix_b_2 = [[ 0.72123296,  0.2553158,  -0.43551614, -0.47430218],
 [ 0.19648728,  0.71249029 , 0.00955327,  0.67354218],
 [ 0.41338402,  0.0366419,   0.89340815, -0.17202586],
 [ 0.51993214 ,-0.65255559, -0.10980116,  0.54017172]]

#####################


def test_bank(parameters_orb_rot, matrix_a, matrix_b):
    # Aqua
    aqua_driver = AquaHDF5Driver(hdf5_input="test_oovqe_h4.hdf5")
    aqua_transform = FermionicTransformation(
        qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER,
        two_qubit_reduction=False,
    )

    aqua_oovqe = OOVQE_Aqua(aqua_transform, solver=None, iterative_oo = False)
    aqua_oovqe._qmolecule = aqua_driver.run()
    aqua_oovqe._orbital_rotation = OR_Aqua(num_qubits=8,
                                            transformation=aqua_transform,
                                            qmolecule=aqua_oovqe._qmolecule)

    aqua_oovqe._orbital_rotation.orbital_rotation_matrix(parameters_orb_rot)
    aqua_oovqe._rotate_orbitals_in_qmolecule(
                aqua_oovqe._qmolecule, aqua_oovqe._orbital_rotation)

    aqua_op, _ = aqua_transform._do_transform(aqua_oovqe._qmolecule)
    aqua_opflow = aqua_op.to_opflow()

    # convert to pauli dict
    aqua_pauli_dict= {}
    for coeff, pauli in aqua_op.paulis:
        if str(pauli) in aqua_pauli_dict.keys():
            raise KeyError
        aqua_pauli_dict[str(pauli)] = coeff.real

    # ---------------------------

    # Nature

    nature_driver = HDF5Driver("test_oovqe_h4.hdf5")
    nature_converter = QubitConverter(JordanWignerMapper())
    nature_problem = CustomProblem(nature_driver)

    oovqe = OrbitalOptimizationVQE(
        qubit_converter=nature_converter
    )
    oovqe.problem = nature_problem
    oovqe.problem.second_q_ops()
    # matrix_a, matrix_b = oovqe.get_matrices(paramters)
    nature_op = oovqe.rotate_orbitals(matrix_a, matrix_b)

    # convert to pauli dict
    nature_pauli_dict = {}
    for p in nature_op:
        for pauli, coeff in p.primitive.label_iter():
            if str(pauli) in nature_pauli_dict.keys():
                raise KeyError
            nature_pauli_dict[str(pauli)] = coeff.real

    # ---------------------------

    # Difference Analysis
    keys_only_in_aqua = set(aqua_pauli_dict.keys()) - set(nature_pauli_dict.keys())
    print(f"{len(keys_only_in_aqua)} Pauli strings only in Aqua: ", keys_only_in_aqua)

    keys_only_in_nature = set(nature_pauli_dict.keys()) - set(aqua_pauli_dict.keys())
    print(f"{len(keys_only_in_nature)} Pauli strings only in Nature: ", keys_only_in_nature)

    diff = nature_pauli_dict.copy()

    for key, value in aqua_pauli_dict.items():
        diff[key] -= value
        if np.isclose(diff[key], 0.0):
            diff.pop(key)

    print(f"{len(diff)} Different keys:")

print("test 1")
test_bank(parameters_1, matrix_a_1, matrix_b_1)

print("test 2")
test_bank(parameters_2, matrix_a_2, matrix_b_2)

