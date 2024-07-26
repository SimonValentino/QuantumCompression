from qiskit import QuantumCircuit, QuantumRegister
import qiskit
import numpy as np
import numpy as np
from qiskit.circuit.library.standard_gates import PhaseGate
from PIL import Image
from arithmetic_operations import adder, muller


# main
img_name = "beaver"
img = Image.open(f"imgs/{img_name}.png").convert("L")
img_arr = np.asarray(img)
print(img_arr)


