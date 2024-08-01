from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np
import numpy as np
from qiskit.circuit.library.standard_gates import PhaseGate
from PIL import Image


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
        return 1 / np.sqrt(2)
    else:
        return 1


# discrete cosine transform on greyscale image
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
                            sum += img_arr[i + p, j + q] * \
                                np.cos((2 * p + 1) * k * np.pi / 16) * \
                                np.cos((2 * q + 1) * l * np.pi / 16)
                    sum *= 1 / 4 * c(k) * c(l)
                    dct[i + k, j + l] = sum

    return dct


# inverse discrete cosine transform on dtc coefficients
def idct(dct_coefficients):
    # loop through 8x8 blocks
    m, n = dct_coefficients.shape
    idct = np.zeros(dct_coefficients.shape)
    for i in range(0, m, 8):
        for j in range(0, n, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    sum = 0
                    for p in range(0, 8):
                        for q in range(0, 8):
                            sum += c(p) * c(q) * dct_coefficients[i + p, j + q] * \
                                np.cos((2 * k + 1) * p * np.pi / 16) * \
                                np.cos((2 * l + 1) * q * np.pi / 16)
                    sum *= 1 / 4
                    idct[i + k, j + l] = sum

    return idct


# puts dtc coefficients trough quantization matrix
def quantize(dct_coefficients, quantization_matrix):
    m, n = dct_coefficients.shape
    quantized = np.zeros(dct_coefficients.shape)
    for i in range(0, m, 8):
        for j in range(0, n, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    quantized[i + k, j + l] = np.round(
                        dct_coefficients[i + k, j + l] / quantization_matrix[k][l])

    return quantized


def dequantize(quantized_arr, quantization_matrix):
    m, n = quantized_arr.shape
    dequantized = np.zeros(quantized_arr.shape)
    for i in range(0, m, 8):
        for j in range(0, n, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    dequantized[i + k, j + l] = quantized_arr[i +
                                                              k, j + l] * quantization_matrix[k][l]

    return dequantized


def apply_UYX(qc, X, Y, coefficient, num_qubits, q):
    # Convert X, Y to binary
    bin_X = format(X, '0{}b'.format(num_qubits - q))
    bin_Y = format(Y, '0{}b'.format(num_qubits - q))

    # Apply X and Y gates
    for i in range(len(bin_X)):
        if bin_X[i] == '1' and i < num_qubits:
            qc.x(i)

    for j in range(len(bin_Y)):
        if bin_Y[j] == '1' and (j + len(bin_X)) < num_qubits:
            qc.x(j + len(bin_X))

    bin_coefficient = format(coefficient, '0{}b'.format(q))
    # Apply CNOT gates based on the coefficient
    for k in range(q):
        if bin_coefficient[k] == '1' and (k + len(bin_X) + len(bin_Y)) < num_qubits:
            qc.cx(k + len(bin_X) + len(bin_Y), num_qubits - q + k)

    # Reset X and Y gates
    for i in range(len(bin_X)):
        if bin_X[i] == '1' and i < num_qubits:
            qc.x(i)

    for j in range(len(bin_Y)):
        if bin_Y[j] == '1' and (j + len(bin_X)) < num_qubits:
            qc.x(j + len(bin_X))


def encode_quantization_matrix(qc, quant_matrix, xy_reg, quantization_reg):
    for i in range(quant_matrix.shape[0]):
        for j in range(quant_matrix.shape[1]):
            xy = 8 * i + j
            bin_xy = bin(xy)[2:].zfill(6)[::-1]

            quant_val = quant_matrix[i][j]
            bin_quant_val = bin(quant_val)[2:].zfill(7)[::-1]

            control_qubits = [k for k in range(6) if bin_xy[k] == "0"]
            target_qubits = [
                6 + k for k in range(7) if bin_quant_val[k] == "1"]

            for ind in control_qubits:
                qc.x(xy_reg[ind])

            for ind in target_qubits:
                qc.mcx(xy_reg, quantization_reg[ind - 6])

            for ind in control_qubits:
                qc.x(xy_reg[ind])


# Main
img = Image.open(f"imgs/beaver.png").convert("L")
img_arr = np.asarray(img)
print("Original image array:\n", img_arr)


# Step 1: DCT and quantize
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

dct_coefficients = dct(img_arr)
quantized_coefficients = quantize(dct_coefficients, quantization_matrix)
print("Quantized DCT Coefficients:\n", quantized_coefficients)


# Step 2: Store the quantized DCT coefficients into quantum systems
q = 8
n = 8

num_qubits = q + 2 * n

x_reg = QuantumRegister(n)
y_reg = QuantumRegister(n)
dct_reg = QuantumRegister(q)
qc = QuantumCircuit(x_reg, y_reg, dct_reg)

for i in range(2 * n):
    qc.h(i)

coefficients = [
    (i, j, int(quantized_coefficients[i, j]))
    for i in range(quantized_coefficients.shape[0])
    for j in range(quantized_coefficients.shape[1])
]

for X, Y, coefficient in coefficients:
    apply_UYX(qc, X, Y, coefficient, num_qubits, q)


# Step 3: Storing the quantization matrix
xy_reg = QuantumRegister(6)
quantization_reg = QuantumRegister(q - 1)
qc.add_register(xy_reg)
qc.add_register(quantization_reg)

encode_quantization_matrix(qc, quantization_matrix, xy_reg, quantization_reg)


# Step 4 inverse quantization
g = QuantumRegister(1)
qc.add_register(g)

for i in range(3):
    qc.cx(x_reg[i], xy_reg[i])
    qc.cx(y_reg[i], xy_reg[i + 3])

qc.x(xy_reg)
qc.mcx(xy_reg, g)
qc.x(xy_reg)

output = QuantumRegister(2 * q - 2)
qc.add_register(output)

print(dct_reg.size)
print(quantization_reg.size)

qc.compose(muller(dct_reg[:-1], quantization_reg, output))  # needs control on g

qc.mcx(g, dct_reg[-1], output[-1])
