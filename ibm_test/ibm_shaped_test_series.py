
import sys
import os
path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('root_path:' ,path)
sys.path.append(path)

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import IBMQ

from qiskit.transpiler import PassManager
import qiskit

from qiskit.circuit.library.standard_gates.equivalence_library import (StandardEquivalenceLibrary as std_eqlib)

# Transpiler passes
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from colib.rzx_compilerv2 import rzx_compiler,rzx_compiler_stretch
import math
from qiskit import Aer, execute
from qiskit.providers.fake_provider  import FakeOpenPulse2Q,FakePerth,FakeLagos,FakeQuito
import pandas as pd
import sys

backend_name  = 'ibmq_kolkata' #'ibmq_montreal'
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl',project='sys-reserve')
# provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')
chunk_size = 300



gate_backend = provider.get_backend(backend_name)
# gate_backend = FakeQuito()
instruction_schedule_map = gate_backend.defaults().instruction_schedule_map
coupling_map = instruction_schedule_map.qubits_with_instruction('cx')

def get_expectations_from_count(counts, n_wires):
    exps = []
    state = np.zeros(pow(2,n_wires))
    sum = 0
    for key,val in counts.items():
        index = int(key,2)
        state[index] = val
        sum +=val
    state = state/sum
    return state


def get_expectations_from_counts(counts, n_wires):
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


# Compare the schedule durations

d = {'theta': [],'qubit': [], 'stretch_ratio': [],'gate-duration': [],'pulse-duration': [],
     'gate-00count': [],'gate-01count': [],'gate-10count': [],'gate-11count': [],
     'pulse-00count': [],'pulse-01count': [],'pulse-10count': [],'pulse-11count': []}
df = pd.DataFrame(data=d)

qc = QuantumCircuit(2)
qc.measure_all()
sche1 = qiskit.schedule(qc, gate_backend)
meas_duration = sche1.duration




def main(theta,stretch_ratio,qubits=[0,1]):
    print()
    print('theta =',theta,'stretch_ratio = ',stretch_ratio)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.rzx(theta,0,1)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    
    qct = transpile(qc, gate_backend,initial_layout=qubits)
    # print(qct)
    sche1 = qiskit.schedule(qct, gate_backend)
    sim_backend = Aer.get_backend('qasm_simulator')
    job = execute(experiments=qct,backend=sim_backend,shots=8192)
    result = job.result()
    counts1 = result.get_counts()
    state1 = get_expectations_from_count(counts1,2)

    sche2 = rzx_compiler_stretch(gate_backend,qc,stretch_ratio,initial_layout=qubits)
    # sche2.draw().savefig('test_pulse2.png')
    # sys.exit(0)
    # print(qc_pulse_efficient)
    # sche2 = qiskit.schedule(qc_pulse_efficient, gate_backend)
    df.loc[len(df.index)] = [theta,qubits,stretch_ratio,sche1.duration - meas_duration, sche2.duration - meas_duration,
                             state1[0],state1[1],state1[2],state1[3],0,0,0,0]
    
    return sche2


theta = math.pi/8
thetas = list(range(1,9))
thetas = np.array(thetas) * theta

# sr1 = np.array(list(range(9,0,-3))) * (-1.0)/29
# sr2 = np.array(list(range(0,19,3))) * (1.0) /29
# stretch_ratios = np.concatenate((sr1,sr2),axis=0)
# stretch_ratios = stretch_ratios +1
# print(stretch_ratios)

stretch_ratios = list(range(6,16,1))
stretch_ratios = np.array(stretch_ratios) * 0.1

sches =[]
coupling_qubits = [(0,1),(1,4),(4,7),(7,10),(10,12),(12,15),(15,18),(1,0),(4,1),(7,4),(10,7),(12,10),(15,12),(18,15)]
for qubits in coupling_qubits:
    # if qubits[0]>qubits[1]:
        for t in thetas:
            for s in stretch_ratios:
                sches.append(main(t,s,qubits)) 


df.to_excel('excels/' +  backend_name +'_rzx_0420_3.xlsx')



split_schedules = [sches[i:i + chunk_size] for i in range(0, len(sches), chunk_size)]

sum_counts = []
for schedules in split_schedules:
    job = execute(experiments=schedules,
                  backend=gate_backend,
                  shots=8192,
                  )
    result = job.result()
    counts = result.get_counts()
    sum_counts.extend(counts)

state2 = get_expectations_from_counts(sum_counts,2)
state2 = np.array(state2)
print(state2.shape)
np.save('excels/' + backend_name +'_state_0420_3',state2)

df['pulse-00count'] = state2[:,0]
df['pulse-01count'] = state2[:,1]
df['pulse-10count'] = state2[:,2]
df['pulse-11count'] = state2[:,3]
    
df.to_excel('excels/'+ backend_name +'_rzx_0420_3.xlsx')