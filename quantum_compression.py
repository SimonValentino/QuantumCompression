from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister
import qiskit
import numpy as np
from qiskit.circuit.library import QFT
import numpy as np


# computes a + b into b
# a and b have to have the same size n
def adder(a: QuantumRegister, b: QuantumRegister):
    n = len(a)
    
    qc = QuantumCircuit(a, b)
    
    qc.append(QFT(n).to_gate(), b)
    
    for j in range(n):
        for k in range(j + 1):
            qc.cp(2 * np.pi / (2 ** (j - k + 1)), a[k], b[j])
    
    qc.append(QFT(n).inverse().to_gate(), b)
    
    return qc


def muller(a: QuantumRegister, b: QuantumRegister, output: QuantumRegister):
    pass
