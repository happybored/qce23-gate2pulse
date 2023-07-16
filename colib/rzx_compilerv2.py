from qiskit import QuantumCircuit, transpile, IBMQ
from qiskit.transpiler import PassManager
import qiskit

import numpy as np
from qiskit.circuit.library.standard_gates.equivalence_library import (StandardEquivalenceLibrary as std_eqlib)

# Transpiler passes
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from ibm_modified.my_rzx_builder import RZXCalibrationBuilderNoEcho,RZXCalibrationBuilder
from ibm_modified.my_rzx_scale_amp_builder import RZXStretchCalibrationBuilder,RZXStretchCalibrationBuilderNoEcho
from ibm_modified.my_rzx_scale_amp_builder2 import RZXStretchCalibrationBuilder2,RZXStretchCalibrationBuilderNoEcho2


# from ibm_modified.my_echo_rzx_weyl_decomposition import (EchoRZXWeylDecomposition)
from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import EchoRZXWeylDecomposition
# New transpiler pass
import math
from qiskit import Aer, execute



def rzx_compiler_stretch2(backend,circuit,stretch_ratio = 1, version = 'v1',initial_layout =None,return_circ = False):
     
    rzx_basis = ["rzx", "rz", "x", "sx",'id']
    if version == 'v1':
        instruction_schedule_map = backend.defaults().instruction_schedule_map
        coupling_map = backend.configuration().coupling_map
    elif version == 'v2':
        instruction_schedule_map = backend.instruction_schedule_map
        coupling_map = backend.coupling_map
    pm2 = PassManager(
        [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=rzx_basis),
            EchoRZXWeylDecomposition(instruction_schedule_map),
            RZXStretchCalibrationBuilderNoEcho2(stretch_ratio,instruction_schedule_map),
            UnrollCustomDefinitions(std_eqlib, rzx_basis),
            BasisTranslator(std_eqlib, rzx_basis),
            Optimize1qGatesDecomposition(rzx_basis),
        ]
    )
    tc = transpile(circuit,coupling_map= coupling_map,optimization_level=1,seed_transpiler=42,initial_layout=initial_layout)
    tc = pm2.run(tc)
    if return_circ:
        return tc
    sche = qiskit.schedule(tc, backend)
    return sche




def rzx_compiler_stretch(backend,circuit,stretch_ratio = 1, version = 'v1',initial_layout =None,return_circ = False):
     
    rzx_basis = ["rzx", "rz", "x", "sx",'id']
    if version == 'v1':
        instruction_schedule_map = backend.defaults().instruction_schedule_map
        coupling_map = backend.configuration().coupling_map
    elif version == 'v2':
        instruction_schedule_map = backend.instruction_schedule_map
        coupling_map = backend.coupling_map
    pm2 = PassManager(
        [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=rzx_basis),
            EchoRZXWeylDecomposition(instruction_schedule_map),
            RZXStretchCalibrationBuilderNoEcho(stretch_ratio,instruction_schedule_map),
            UnrollCustomDefinitions(std_eqlib, rzx_basis),
            BasisTranslator(std_eqlib, rzx_basis),
            Optimize1qGatesDecomposition(rzx_basis),
        ]
    )
    tc = transpile(circuit,coupling_map= coupling_map,optimization_level=1,seed_transpiler=42,initial_layout=initial_layout)
    tc = pm2.run(tc)
    if return_circ:
        return tc
    sche = qiskit.schedule(tc, backend)
    return sche




def rzx_compiler(backend,circuit,version = 'v1',return_circ = False):
     
    rzx_basis = ["rzx", "rz", "x", "sx",'id']
    if version == 'v1':
        instruction_schedule_map = backend.defaults().instruction_schedule_map
        coupling_map = backend.configuration().coupling_map
    elif version == 'v2':
        instruction_schedule_map = backend.instruction_schedule_map
        coupling_map = backend.coupling_map
    pm2 = PassManager(
        [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=rzx_basis),
            EchoRZXWeylDecomposition(instruction_schedule_map),
            RZXCalibrationBuilderNoEcho(instruction_schedule_map),
            UnrollCustomDefinitions(std_eqlib, rzx_basis),
            BasisTranslator(std_eqlib, rzx_basis),
            Optimize1qGatesDecomposition(rzx_basis),
        ]
    )
    tc = transpile(circuit,coupling_map= coupling_map,optimization_level=1,seed_transpiler=42)
    tc = pm2.run(tc)
    if return_circ:
        return tc
    sche = qiskit.schedule(tc, backend)
    return sche
