from qiskit import IBMQ
from qiskit import QuantumCircuit, transpile, assemble, execute
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
IBMQ.save_account('') #Set to Madhav's token rn
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibm_sherbrooke')
#Random Circuit for Demo
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
#Makes it readable to the QC
transpiled_qc = transpile(qc, backend)
# Formats to be used correctly with the backend api
qobj = assemble(transpiled_qc)
#Runs circuit
job = backend.run(qobj)
job_monitor(job)
result = job.result()
#Plots Results
counts = result.get_counts()
plot_histogram(counts)

#Lists all avaliable backends to use
# backends = provider.backends()
# for backend in backends:
#     print(backend.name())