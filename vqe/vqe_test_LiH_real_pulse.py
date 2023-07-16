import sys
import os

path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('root_path:' ,path)
sys.path.append(path)
from ibm_modified.my_vqe import VQE

import matplotlib.pyplot as plt
import qiskit
import qiskit_nature
import qiskit_nature.problems.second_quantization
import qiskit_nature.drivers.second_quantization
import qiskit_nature.transformers.second_quantization.electronic
import qiskit_nature.algorithms
from qiskit_nature.drivers import Molecule
import numpy as np
from colib.rzx_compilerv2 import rzx_compiler,rzx_compiler_stretch
from qiskit import Aer, execute

from qiskit.circuit.library import EfficientSU2,TwoLocal
import sys
from qiskit.providers.fake_provider  import FakeMontreal
from qiskit.primitives import Estimator, Sampler
from qiskit import QuantumCircuit, transpile, IBMQ
import random
from qiskit.algorithms import optimizers
import copy
from qiskit.providers.ibmq.job import job_monitor
rotations = ['rx']
entangles = ['rzx']
chunk_size = 300
import math

def get_states_from_counts(counts, n_wires):
    states = []
    
    for count in counts:
        sum = 0
        state = np.zeros(pow(2,n_wires))
        for key,val in count.items():
            index = int(key,2)
            state[index] = val
            sum +=val
        state = state/sum
        states.append(state)
    return states


def get_expectations_from_counts(counts, n_wires):
    exps = []
    if isinstance(counts, dict):
        counts = [counts]
    for count in counts:
        ctr_one = [0] * n_wires
        total_shots = 0
        for k, v in count.items():
            for wire in range(n_wires):
                if k[wire] == "1":
                    ctr_one[wire] += v
            total_shots += v
        prob_one = np.array(ctr_one) / total_shots
        exp = np.flip(-1 * prob_one + 1 * (1 - prob_one))
        exps.append(exp)

    res = np.stack(exps)
    # res = np.cumprod(res,1)[:,-1]
    return res

def get_qubit_op(molecule,remove_orbitals):
    driver = qiskit_nature.drivers.second_quantization.ElectronicStructureMoleculeDriver(
        molecule=molecule,
        basis="sto3g",
        driver_type=qiskit_nature.drivers.second_quantization.ElectronicStructureDriverType.PYSCF)

    # Define Problem, Use freeze core approximation, remove orbitals.
    problem = qiskit_nature.problems.second_quantization.ElectronicStructureProblem(driver,remove_orbitals)

    second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
    num_spin_orbitals = problem.num_spin_orbitals
    num_particles = problem.num_particles

    mapper = qiskit_nature.mappers.second_quantization.ParityMapper()  # Set Mapper
    # print(second_q_ops)
    hamiltonian = second_q_ops['ElectronicEnergy']  # Set Hamiltonian
    # sys.exit(0)
    # Do two qubit reduction
    converter = qiskit_nature.converters.second_quantization.QubitConverter(mapper,two_qubit_reduction=True)
    reducer = qiskit.opflow.TwoQubitReduction(num_particles)
    qubit_op = converter.convert(hamiltonian)
    qubit_op = reducer.convert(qubit_op)

    return qubit_op, num_particles, num_spin_orbitals, problem, converter

def exact_solver(problem, converter):
    solver = qiskit_nature.algorithms.NumPyMinimumEigensolverFactory()
    calc = qiskit_nature.algorithms.GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    return result



backend = qiskit.BasicAer.get_backend("statevector_simulator")
backend_name  = 'ibmq_kolkata' #'ibmq_montreal'
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl',project='sys-reserve')
gate_backend = provider.get_backend(backend_name)
# gate_backend = qiskit.BasicAer.get_backend('qasm_simulator')

distances = np.arange(1.2,3.2, 0.4)
exact_energies = []
vqe_energies = []

for dist in distances:


    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    # torch.manual_seed(seed)

    # Define Molecule
    molecule = Molecule(
        # Coordinates in Angstrom
        geometry=[
            ["Li", [0.0, 0.0, 0.0] ],
            ["H", [dist, 0.0, 0.0] ]
        ],
        multiplicity=1,  # = 2*spin + 1
        charge=0,
    )
    (qubit_op, num_particles, num_spin_orbitals,problem, converter) = get_qubit_op(molecule,[qiskit_nature.transformers.second_quantization.electronic.FreezeCoreTransformer(freeze_core=True,remove_orbitals=[-3,-2])])

    qubit_op_copy = copy.deepcopy(qubit_op) 
    
    result = exact_solver(problem,converter)
    exact_energies.append(result.total_energies[0].real)

    

    # print(qubit_op.num_qubits)


    init_state = qiskit_nature.circuit.library.HartreeFock(num_spin_orbitals, num_particles, converter)

    

    var_form = TwoLocal(qubit_op.num_qubits, rotations, entangles, 'linear', reps=2)

    # sampler = Sampler()
    # fidelity = qiskit.algorithms.optimizers.QNSPSA.get_fidelity(var_form, sampler)

    # optimizer = qiskit.algorithms.optimizers.QNSPSA(fidelity, maxiter=300)
    optimizer = optimizers.COBYLA(maxiter=2000)

    vqe = VQE(var_form, optimizer, quantum_instance=backend)

    # print(qubit_op_copy)
    vqe_calc = vqe.compute_minimum_cost_function(qubit_op)
    # vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)

    

    #result
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real

    # print(vqe.ansatz)
    # print(vqe_calc.optimal_point)
    
    

    # vqe.quantum_instance = gate_backend
    # eigenvalue = vqe.get_energy_evaluation_from_param(qubit_op,vqe_calc.optimal_point)
    # vqe_calc.eigenvalue = eigenvalue
    # vqe_result2 = problem.interpret(vqe_calc).total_energies[0].real
    # print(f"Gate-noise VQE Result: {vqe_result2:.5f} ",
    #       f"Exact Energy: {exact_energies[-1]:.5f} ")
    
    # dt = gate_backend.dt
    # print(dt)
    circ = vqe.ansatz.assign_parameters(vqe_calc.optimal_point,inplace= False)
    qct = transpile(circ, gate_backend,seed_transpiler=42)
    sche1 = qiskit.schedule(qct, gate_backend) #original 
    sche2 = rzx_compiler_stretch(gate_backend,circ,1)
    sche3 = rzx_compiler_stretch(gate_backend,circ,0.9)




    # vqe_energies.append(vqe_result)
    print(f"gate-based 'duration:{sche1.duration} ",
          f"pulse-based 'duration:{sche2.duration} ",
          f"scaled pulse-based 'duration:{sche3.duration} ",
          f"Interatomic Distance: {np.round(dist, 2)} ",
          f"VQE eigenvalue: {vqe_calc.eigenvalue:.5f} ",
          f"VQE Result: {vqe_result:.5f} ",
          f"Exact Energy: {exact_energies[-1]:.5f} ")
    
    
    weight,ops,sampled_circuit = vqe.construct_circuit2(vqe_calc.optimal_point,operator= qubit_op)


    transpiled_circs = transpile(sampled_circuit, gate_backend,seed_transpiler=42)
    sches2 = rzx_compiler_stretch(gate_backend,sampled_circuit,1)
    sches3 = rzx_compiler_stretch(gate_backend,sampled_circuit,0.9)
    # sches = 

    
    sum_counts = []
    job = execute(experiments=transpiled_circs,
                  backend=gate_backend,
                  shots=8192,
                  )
    # job_monitor(job,interval=1)
    
    result = job.result()
    counts = result.get_counts()
    sum_counts.extend(counts)    
    state2 = get_expectations_from_counts(sum_counts,qubit_op.num_qubits)
    eigen_value = 0
    for i in range(state2.shape[0]):
        for j in range(len(ops[i])):
            if ops[i][j]== 'I':
                state2[i][7-j] = 1 
    state2 = np.cumprod(state2,axis=1)[:,-1]
    for i in range(state2.shape[0]):
        eigen_value += state2[i] *weight[i]
    print(eigen_value)
    vqe_calc.eigenvalue =  eigen_value
    vqe_result2 = problem.interpret(vqe_calc).total_energies[0].real

    print(f"Gate Noise Result: {vqe_result2:.5f} ",
          f"Exact Energy: {exact_energies[-1]:.5f} ")
    

    sum_counts = []
    job = execute(experiments=sches2,
                  backend=gate_backend,
                  shots=8192,
                  )
    # job_monitor(job,interval=1)
    
    result = job.result()
    counts = result.get_counts()
    sum_counts.extend(counts)    
    state2 = get_expectations_from_counts(sum_counts,qubit_op.num_qubits)
    eigen_value = 0
    for i in range(state2.shape[0]):
        for j in range(len(ops[i])):
            if ops[i][j]== 'I':
                state2[i][7-j] = 1 
    state2 = np.cumprod(state2,axis=1)[:,-1]
    for i in range(state2.shape[0]):
        eigen_value += state2[i] *weight[i]
    print(eigen_value)
    vqe_calc.eigenvalue =  eigen_value
    vqe_result2 = problem.interpret(vqe_calc).total_energies[0].real

    print(f"Pulse Noise Result: {vqe_result2:.5f} ",
          f"Exact Energy: {exact_energies[-1]:.5f} ")


    sum_counts = []
    job = execute(experiments=sches3,
                  backend=gate_backend,
                  shots=8192,
                  )
    # job_monitor(job,interval=1)
    
    result = job.result()
    counts = result.get_counts()
    sum_counts.extend(counts)    
    state2 = get_expectations_from_counts(sum_counts,qubit_op.num_qubits)
    eigen_value = 0
    for i in range(state2.shape[0]):
        for j in range(len(ops[i])):
            if ops[i][j]== 'I':
                state2[i][7-j] = 1 
    state2 = np.cumprod(state2,axis=1)[:,-1]
    for i in range(state2.shape[0]):
        eigen_value += state2[i] *weight[i]
    print(eigen_value)
    vqe_calc.eigenvalue =  eigen_value
    vqe_result2 = problem.interpret(vqe_calc).total_energies[0].real

    print(f"Scaled Pulse Noise Result: {vqe_result2:.5f} ",
          f"Exact Energy: {exact_energies[-1]:.5f} ")
    
    print()
    
print("All energies have been calculated")


# plt.plot(distances, exact_energies, label="Exact Energy")
# plt.plot(distances, vqe_energies, label="VQE Energy")
# plt.xlabel('Atomic distance (Angstrom)')
# plt.ylabel('Energy')
# plt.legend()
# plt.show()