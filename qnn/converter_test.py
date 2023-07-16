import torch
import argparse

import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np
from q_layers import QLayer_RZX



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')

    args = parser.parse_args()


    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_model = QLayer_RZX()
    # convert the tq module to qiskit and draw
    from torchquantum.plugins import tq2qiskit, qiskit2tq
    circ = tq2qiskit(tq.QuantumDevice(n_wires=q_model.n_wires), q_model, draw=True)

    # convert the QiskitCircuit to tq module
    q_model_back = qiskit2tq(circ)
    print(q_model_back)


if __name__ == '__main__':
    main()