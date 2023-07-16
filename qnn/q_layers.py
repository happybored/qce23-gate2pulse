from __future__ import print_function

import torchquantum as tq



class QLayer_RZX(tq.QuantumModule):
    def __init__(self,n_wire =4):
        super().__init__()
        self.n_wires = n_wire
        self.layer_indexs  = dict()
        self.RZs1 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZs2 = tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs1 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RXs2 = tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RYs1 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RYs2 = tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,has_params=True,trainable=True)
        self.RZX1 = tq.Op2QAllLayer(op=tq.RZX,n_wires=self.n_wires,has_params=True,trainable=True,circular =False) #Op2QAllLayer
        self.RZX2 = tq.Op2QAllLayer(op=tq.RZX,n_wires=self.n_wires,has_params=True,trainable=True,circular =False) #Op2QAllLayer
        self.RZX3 = tq.Op2QAllLayer(op=tq.RZX,n_wires=self.n_wires,has_params=True,trainable=True,circular =False) #Op2QAllLayer

        # self.CRZs3 = tq.Op2QAllLayer(op=tq.CRZ,n_wires=3,has_params=True,trainable=True,circular =True)
        # self.hadmard = tq.Hadamard(n_wires=4,wires=[0, 1,2,3])
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.RZs1(self.q_device)
        self.RZX1(self.q_device)
        self.RZs2(self.q_device)

        self.RXs1(self.q_device)
        self.RZX2(self.q_device)
        self.RXs2(self.q_device)

        self.RYs1(self.q_device)
        self.RZX3(self.q_device)
        self.RYs2(self.q_device)