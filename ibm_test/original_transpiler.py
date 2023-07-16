
import sys
import os
path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('root_path:' ,path)
sys.path.append(path)

from qiskit import QuantumCircuit
from qiskit import transpile, schedule
from qiskit import pulse
from qiskit.providers.fake_provider import FakeHanoi,FakeKolkata
from qiskit.pulse import Schedule
from colib.rzx_compilerv2 import rzx_compiler
import qiskit

import math
backend = FakeKolkata()

circ = QuantumCircuit(2, 2)
# circ.crx(5*math.pi/8,1,0)
# circ.x(0)
# circ.rzx(6*math.pi/8,1,0)
circ.cx(0,1)
print(circ)


transpiled_circ = transpile(circ,backend=backend) 
print(transpiled_circ)
pulse = qiskit.schedule(transpiled_circ, backend)
# print(transpiled_circ)
# print('cx by cx',pulse.duration)
# pulse2 = rzx_compiler(backend,circ,return_circ=True)
# print(pulse2)
# print('cx by rzx',pulse2.duration)

pulse.draw().savefig('cx-cx.jpg')
pulse.draw().savefig('cx-cx.eps')