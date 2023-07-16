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
from qiskit.circuit.library import EfficientSU2,TwoLocal
import sys
from qiskit.providers.fake_provider  import FakeQuito,FakeCairoV2,FakeParisV2
from qiskit.primitives import Estimator, Sampler
from qiskit import QuantumCircuit, transpile, IBMQ
from colib.rzx_compilerv2 import rzx_compiler
import random
from qiskit.algorithms import optimizers

rotations = ['rx']
entangles = ['rzx']





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
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc517')
gate_backend = provider.get_backend('ibmq_montreal')



distances = np.arange(0.4,4.4, 0.4)
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

    vqe_calc = vqe.compute_minimum_cost_function(qubit_op)

    # print(vqe.ansatz)
    # print(vqe_calc.optimal_point)
    circ = vqe.ansatz.assign_parameters(vqe_calc.optimal_point,inplace= False)

    
    # dt = gate_backend.dt
    # print(dt)
    qct = transpile(circ, gate_backend,seed_transpiler=42)
    sche1 = qiskit.schedule(qct, gate_backend)
    # print('gate-based ','duration:',sche1.duration)
    sche2 = rzx_compiler(gate_backend,circ)


    #result
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    vqe_energies.append(vqe_result)
    print(f"gate-based 'duration:{sche1.duration} ",
          f"pulse-based 'duration:{sche2.duration} ",
          f"Interatomic Distance: {np.round(dist, 2)} ",
          f"VQE Result: {vqe_result:.5f} ",
          f"Exact Energy: {exact_energies[-1]:.5f} ")
    # print()

print("All energies have been calculated")


# plt.plot(distances, exact_energies, label="Exact Energy")
# plt.plot(distances, vqe_energies, label="VQE Energy")
# plt.xlabel('Atomic distance (Angstrom)')
# plt.ylabel('Energy')
# plt.legend()
# plt.show()