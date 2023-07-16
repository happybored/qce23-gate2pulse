import numpy as np

import sys
import os
path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('root_path:' ,path)
sys.path.append(path)

import cma
from qiskit import qpy
import math
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2,TwoLocal
from qiskit import IBMQ
from colib.rzx_compilerv2 import rzx_compiler_stretch,rzx_compiler_stretch2
from qiskit.providers.fake_provider  import FakeOpenPulse2Q,FakePerth,FakeLagos,FakeQuito
import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit,DAGOpNode
import pandas as pd
import matplotlib as plt
backend_name  = 'ibmq_kolkata' #'ibmq_montreal'
chunk_size = 300

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl',project='sys-reserve')
gate_backend = provider.get_backend(backend_name)
instruction_schedule_map = gate_backend.defaults().instruction_schedule_map
coupling_map = instruction_schedule_map.qubits_with_instruction('cx')

# gate_backend = FakeLagos()

def get_circuit_from_path(path):
    with open(path, 'rb') as fd:
        circuit = qpy.load(fd)
    return circuit

def get_hardware_efficient_circuit(qubits,rotations,entangles,reps):
    np.random.seed(42)
    qc = QuantumCircuit(qubits)
    var_form = TwoLocal(qubits, rotations, entangles, 'linear', reps=reps)
    var_form.assign_parameters(np.random.randn(var_form.num_parameters)*math.pi,inplace= True)
    qc = qc.compose(var_form)
    qc.measure_all()
    return qc


def fitting(X,k1,k2,b1):
    dsr,theta = X
    return np.cos(theta)/2+0.5 + (np.sin(theta)/2 * k1*np.sin((k2*(dsr-1)+b1))*np.sign(theta - 1.7))

def get_error_rate(X,k1,k2,b1):
    dsr,theta = X
    return np.abs(np.sin(theta)/2 * k1*np.sin((k2*(dsr-1)+b1))*np.sign(theta - 1.7))

def callback(x):
    print('loss =',x.result[0])

def get_dsr(x):
    dsrs =dict()
    coupling_qubits = [(0,1),(1,4),(4,7),(7,10),(10,12),(12,15),(15,18),
                       (1,0),(4,1),(7,4),(10,7),(12,10),(15,12),(18,15)]
    for i in range(len(x)):
        dsrs[str(coupling_qubits[i])] = x[i]
        dsrs[str(coupling_qubits[i + len(x)])] = x[i]
    return dsrs




def my_func(x,LUT,circuit,beta,ori_duration):
    # dsrs = get_dsr(x)
    dsrs = x[0]
    # print(dsrs)
    circ = rzx_compiler_stretch2(gate_backend,circuit,dsrs,initial_layout=[0,1,4,7,10,12,15,18],return_circ=True)
    sche = qiskit.schedule(circ, gate_backend)
    new_duration = sche.duration
    norm_duration =new_duration*1.0/ori_duration
    

    # traverse all gate to get error rate
    sum_error_rate =0
    dag = circuit_to_dag(circ)
    two_qubit_ops = dag.two_qubit_ops()
    error_rates = dict()
    
    for two_qubit_op in two_qubit_ops:
        q0 =two_qubit_op.qargs[0].index
        q1 =two_qubit_op.qargs[1].index
        theta = two_qubit_op.op.params[0]
        coffs =LUT[str((q0,q1))]
        # error_rate = get_error_rate((dsrs[str((q0,q1))],theta),coffs['k1'],coffs['k2'],coffs['b1'])
        error_rate = get_error_rate((dsrs,theta),coffs['k1'],coffs['k2'],coffs['b1'])
        if str((q0,q1)) not in error_rates.keys():
            error_rates[str((q0,q1))] = [error_rate]
        else:
            error_rates[str((q0,q1))].append(error_rate)
    for key,val in error_rates.items():
        sum_error_rate += max(val)
    sum_error_rate = sum_error_rate/len(error_rates.keys())

    
    loss = norm_duration +beta * sum_error_rate
    # print('duration={},max_error={},dsr={},loss={}'.format(norm_duration,beta * sum_error_rate,dsr,loss))
    return loss



# Define the initial guess
# x0 = np.ones(7)*0.9
x0 =[1.2,1.2]
df = pd.read_excel('excels/ibmq_kolkata_rzx_0420_3_info.xlsx')
LUT = dict()
for index, row in df.iterrows():
    LUT[row['qubit']] = {'k1':row['k1'],'k2':row['k2'],'b1':row['b1']}


circuit = get_hardware_efficient_circuit(8,['rx'],['rzx'],3)
circ = rzx_compiler_stretch2(gate_backend,circuit,1.0,initial_layout=[0,1,4,7,10,12,15,18],return_circ=True) 
ori_duration = qiskit.schedule(circ, gate_backend).duration


# Set up the options for the optimizer
opts = cma.CMAOptions()
opts.set("bounds", [0.5,1.5]) # set bounds for the variables
opts.set("maxiter", 10) # set maximum number of iterations
opts.set("popsize", 4) # set the population size

# Use the fmin2 function to minimize the function
res = cma.fmin2(my_func, x0, 0.1,args=(LUT,circuit,10,ori_duration), options=opts,callback=callback)


# initialize an empty list to store the function values at each iteration


# plot the function value at each iteration
# plt.plot(function_values)
# plt.xlabel('Iteration')
# plt.ylabel('Function Value')
# plt.title('Function Value vs. Iteration')
# plt.show()

# print(res)