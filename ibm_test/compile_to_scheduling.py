import sys
import os
path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('root_path:' ,path)
sys.path.append(path)

from qiskit.providers.fake_provider import FakeBelem
from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import EfficientSU2,TwoLocal
from qiskit.visualization.timeline import draw
import numpy as np
from colib.rzx_compilerv2 import rzx_compiler
import qiskit

circ = QuantumCircuit(3)
rotations = ['rx']
entangles = ['crx']
var_form = TwoLocal(3, rotations, entangles, 'linear', reps=1,insert_barriers=True)
var_form.assign_parameters(np.random.randn(var_form.num_parameters ),inplace= True)
circ = circ.compose(var_form)
backend = FakeBelem()


pulse = rzx_compiler(backend,circ)
pulse.draw().savefig('3pp2.jpg')


