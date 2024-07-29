from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np
import numpy as np
from qiskit.circuit.library.standard_gates import PhaseGate
from PIL import Image

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


def c(u):
    if u == 0:
        return 1/np.sqrt(2)
    else:
        return 1


def dct(img_arr):
    # loop through 8x8 blocks
    m, n = img_arr.shape
    dct = np.zeros(img_arr.shape)
    for i in range(0, m, 8):
        for j in range(0, n, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    sum = 0
                    for p in range(0, 8):
                        for q in range(0, 8):
                            sum += img_arr[i+p, j+q] * \
                                np.cos((2*p+1)*k*np.pi/16) * \
                                np.cos((2*q+1)*l*np.pi/16)
                    sum *= 0.25 * c(k) * c(l)
                    dct[i+k, j+l] = sum


def quantize(dct_coefficients):
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    m, n = dct_coefficients.shape
    quantized = np.zeros(dct_coefficients.shape)
    for i in range(0, m, 8):
        for j in range(0, n, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    quantized[i+k, j+l] = np.round(dct_coefficients[i+k, j+l] / quantization_matrix[k][l])

    return quantized


# main
img_name = "beaver"
img = Image.open(f"imgs/{img_name}.png").convert("L")
img_arr = np.asarray(img)
print(img_arr)
