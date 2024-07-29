# from qiskit import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService, Session,SamplerV2 as Sampler,EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
import numpy as np
# from qiskit.tools.monitor import job_monitor
# service = QiskitRuntimeService(channel="ibm_quantum", token="'edca0bb4e67297b867b8724895bad5ed212b457974416572dc4e5d602c59e8082bf6611fe2313dd69c233eb8649953fe33370abc10392e6e2f6532b91c100d54'")
# service = QiskitRuntimeService()
# service.backends()

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
 
# Load saved credentials
service = QiskitRuntimeService()
#ibm_quantum_service = QiskitRuntimeService(channel="ibm_quantum", token="edca0bb4e67297b867b8724895bad5ed212b457974416572dc4e5d602c59e8082bf6611fe2313dd69c233eb8649953fe33370abc10392e6e2f6532b91c100d54")

#Test circuit preparing the quantum state
theta = Parameter('θ')
circuit = QuantumCircuit(3)
circuit.h(0) # generate superposition
circuit.p(theta, 0) # add quantum phase
circuit.cx(0, 1) # condition 1st qubit on 0th qubit
circuit.cx(0, 2) # condition 2nd qubit on 0th qubit
# The observable to be measured
M1 = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

gr = (np.sqrt(5) + 1) / 2 # golden ratio
thetaa = 0 # lower range of theta
thetab = 2*np.pi # upper range of theta
tol = 1e-1 # tol
backend = service.least_busy(operational=True, simulator=False)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(circuit)
isa_observables = M1.apply_layout(isa_circuit.layout)

# 3. Execute iteratively using the Estimator primitive
with Session(service=service, backend=backend) as session:
    estimator = Estimator(session=session)
    estimator.options.default_precision = 0.03  # Options can be set using auto-complete.
    #next test range
    thetac = thetab - (thetab - thetaa) / gr
    thetad = thetaa + (thetab - thetaa) / gr
    while abs(thetab - thetaa) > tol:
        print(f"max value of M1 is in the range theta = {[thetaa, thetab]}")
        job = estimator.run([(isa_circuit, isa_observables, [[thetac],[thetad]])])
        test = job.result()[0].data.evs
        if test[0] > test[1]:
            thetab = thetad
        else:
            thetaa = thetac
        thetac = thetab - (thetab - thetaa) / gr
        thetad = thetaa + (thetab - thetaa) / gr
  # Final job to evaluate Estimator at midpoint found using golden search method
    theta_mid = (thetab + thetaa) / 2
    job = estimator.run([(isa_circuit, isa_observables, theta_mid)])
    print(f"Session ID is {session.session_id}")
    print(f"Final Job ID is {job.job_id()}")
    print(f"Job result is {job.result()[0].data.evs} at theta = {theta_mid}")        
# pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
# isa_circuit = pm.run(bell)
# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibm_sherbrooke')
# #Random Circuit for Demo
# qc = QuantumCircuit(2, 2)
# qc.h(0)
# qc.cx(0, 1)
# qc.measure([0, 1], [0, 1])
# #Makes it readable to the QC
# transpiled_qc = transpile(qc, backend)
# # Formats to be used correctly with the backend api
# qobj = assemble(transpiled_qc)
# #Runs circuit
# job = backend.run(qobj)
# # job_monitor(job)
# result = job.result()
# #Plots Results
# counts = result.get_counts()
# plot_histogram(counts)

#Lists all avaliable backends to use
# backends = provider.backends()
# for backend in backends:
#     print(backend.name())

