from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.circuit.library.standard_gates import PhaseGate

# Performs Quantum Fourier Transform
# invert to do inverse and reverse to swap the ordering
def qft(n, invert, reverse):
    qc = QuantumCircuit(n)
    for i in reversed(range(n)):
        qc.h(i)
        for j in reversed(range(i)):
            qc.cp(np.pi * (2.0 ** (j - i)), j, i)

    if reverse:
        for i in range(n // 2):
            qc.swap(i, n - i - 1)

    if invert:
        qc = qc.inverse()

    return qc


# computes a + b into b
# adder shown in this paper https://arxiv.org/pdf/quant-ph/0008033
def adder(a: QuantumRegister, b: QuantumRegister):
    n = len(a)

    qc = QuantumCircuit(a, b)

    qc.append(qft(n, reverse=False).to_gate(), b)

    for j in range(n):
        for k in range(j + 1):
            qc.cp(2 * np.pi / (2 ** (j - k + 1)), a[k], b[j])

    qc.append(qft(n, invert=False, reverse=False).to_gate(), b)

    return qc


# multiplies a and b into output, which is initialized to 0s
# multiplier shown in this paper https://arxiv.org/pdf/1411.5949
def muller(a: QuantumRegister, b: QuantumRegister, output: QuantumRegister):
    n = len(a)

    qc = QuantumCircuit(a, b, output)

    firstqft = qft(2 * n, False, False).to_gate()
    firstqft.name = 'QFT'

    qc.append(firstqft, output)

    for j in range(1, n + 1):
        for i in range(1, n + 1):
            for k in range(1, 2 * n + 1):
                rot = (2 * np.pi) / (2 ** (i + j + k - 2 * n))
                qc.append(PhaseGate(rot).control(2),
                          [a[n - j], b[n - i], output[k - 1]],)

    secondqft = qft(2 * n, True, False).to_gate()
    secondqft.name = 'iQFT'

    qc.append(secondqft, output)

    return qc
