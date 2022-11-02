How to create a ``FermionicOp``
===============================

>>> from qiskit_nature.second_q.operators import FermionicOp
>>> creation = FermionicOp({"+_0": 1})
>>> print(creation)
Fermionic Operator
number spin orbitals=1, number terms=1
  1 * ( +_0 )

>>> from qiskit_nature.second_q.operators import FermionicOp
>>> annihilation = FermionicOp({"-_0": 1})
>>> print(annihilation)
Fermionic Operator
number spin orbitals=1, number terms=1
  1 * ( -_0 )

>>> from qiskit_nature.second_q.operators import FermionicOp
>>> number = FermionicOp({"+_0 -_0": 1})
>>> print(number)
Fermionic Operator
number spin orbitals=1, number terms=1
  1 * ( +_0 -_0 )

>>> from qiskit_nature.second_q.operators import FermionicOp
>>> emptiness = FermionicOp({"-_0 +_0": 1})
>>> print(emptiness)
Fermionic Operator
number spin orbitals=1, number terms=1
  1 * ( -_0 +_0 )

>>> from qiskit_nature.second_q.operators import FermionicOp
>>> hopping = FermionicOp({"-_0 +_1": 1})
>>> print(hopping)
Fermionic Operator
number spin orbitals=2, number terms=1
  1 * ( -_0 +_1 )
