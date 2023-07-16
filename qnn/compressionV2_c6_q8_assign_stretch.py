import sys
import os

path =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print('root_path:' ,path)
sys.path.append(path)

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import math
import torchquantum as tq
import torchquantum.functional as tqf
import sys
from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from qiskit import QuantumCircuit, transpile, IBMQ
import qiskit

# from torchquantum.plugins import QiskitProcessor
from modified.qiskit_processor import QiskitProcessor
from q_layers import QLayer4,QLayer_RZX
import random
import numpy as np
import pandas as pd
import warnings
from torchquantum.plugins import tq2qiskit, qiskit2tq
from colib.rzx_compilerv2 import rzx_compiler_stretch2


from qiskit.providers.fake_provider  import FakeQuito


warnings.filterwarnings("ignore")


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = QLayer_RZX(wire)
            self.layer2 = QLayer_RZX(wire)
            self.layer3 = QLayer_RZX(wire)
            self.layer4 = QLayer_RZX(wire)
            self.layer5 = QLayer_RZX(wire)
            self.layer6 = QLayer_RZX(wire)
            self.layer7 = QLayer_RZX(wire)
            self.layer8 = QLayer_RZX(wire)
            self.layer9 = QLayer_RZX(wire)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.layer1(self.q_device)
            self.layer2(self.q_device)
            self.layer3(self.q_device)
            self.layer4(self.q_device)
            self.layer5(self.q_device)
            self.layer6(self.q_device)
            self.layer7(self.q_device)
            self.layer8(self.q_device)
            self.layer9(self.q_device)

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['6x6_ryrzrx'])

        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 4).view(bsz, 36)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        # print(x.shape)
        x = x[:,0:6].reshape(bsz, 6, 1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


def train(dataflow, model, device, optimizer,mask):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs)
        closs = F.nll_loss(outputs, targets)
        rloss1 =0
        rloss2 =0
        i =0 
        for param in model.parameters():
            base = torch.round(param/math.pi)* math.pi
            regu = torch.abs(param-base) > math.pi/4
            # print(regu)
            rloss1 =rloss1 + torch.sum(torch.abs(param-base)) *mask[i] #* regu
            # rloss2 =rloss1 + torch.sum(torch.abs(param)) *mask[i]
            i = i+1
        # print(rloss)
        loss = closs  + rloss1 *0.0005
        # sys.exit(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')





def valid(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    print(f"{split} set accuracy: {accuracy}")
    return accuracy


def get_stretch_ratios(path,method = 'cross'):
    df = pd.read_excel(path)
    stretch_ratios = dict()
    qubits = df['qubit'].unique()
    thetas = df['theta'].unique()
    for qubit in qubits:
        sub_ratio =  dict()
        for theta in thetas:
            sub_df = df[(df['qubit'] ==qubit)&(df['theta'] ==theta)]
            sub_ratio[theta] = float(sub_df[method])
        stretch_ratios[qubit]=sub_ratio
    return stretch_ratios



def main(backend_name):
    

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    backend = provider.get_backend(backend_name)

    dataset = MNIST(
        root='./mnist_data',
        center_crop = 24,
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0,1,2,3,6,9],
        n_test_samples=200,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        if split == 'test':
            bs = 100
        else:
            bs =64
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=bs,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)



    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    name_list = []
    for name,param in model.q_layer.named_parameters():
        name_list.append(name.split('.')[4].split('_')[0])
    mask = np.array(name_list) == 'RZX'
    print(name_list)
    print(mask)

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    if train_flag:
        best_acc = 0
        for epoch in range(1, n_epochs + 1):
            # train
            print(f"Epoch {epoch}:")
            train(dataflow, model, device, optimizer,mask)
            # print(optimizer.param_groups[0]['lr'])
    
            # valid
            acc = valid(dataflow, 'valid', model, device)
            scheduler.step()

            if acc >best_acc:
                best_acc = acc
                torch.save(model,model_path)
                print(f'best acc = {best_acc}, saved!',)
    else:
        model = torch.load(model_path)


    

    #ADMM Training
    print(f"\nTest with Torchquantun")
    valid(dataflow, 'test', model, device, qiskit=False)

    
    # print(f"\nTest with Qiskit Simulator")
    # processor_simulation = QiskitProcessor(use_real_qc=False,backend_name=backend_name,hub='ibm-q-ornl', group='ornl', project='csc517')
    # model.set_qiskit_processor(processor_simulation)
    # valid(dataflow, 'test', model, device, qiskit=True)



    circ = tq2qiskit(tq.QuantumDevice(n_wires=8), model.q_layer, draw=True)

    
    stretch_ratios = get_stretch_ratios(excel_path,method='cross')
    print(stretch_ratios)
    sche2 = rzx_compiler_stretch2(backend,circ,0.9)
    print("pulse-based ",'duration:',sche2.duration) 
   
    print(f"\nTest with Pulse Schedule")
    processor = QiskitProcessor(use_real_qc=True,backend_name=backend_name,hub='ibm-q-ornl', group='ornl', project='sys-reserve',use_pulse= True,max_experiments=100)
    model.set_qiskit_processor(processor)
    valid(dataflow, 'test', model, device, qiskit=True)  





parser = argparse.ArgumentParser()
parser.add_argument('--static', action='store_true', help='compute with '
                                                          'static mode')
parser.add_argument('--epochs', type=int, default=8,
                    help='number of training epochs')

parser.add_argument('--train', type=bool, default=False,
                    help='number of training epochs')

parser.add_argument('--mpath', type=str, default= 'model/', help='load model')
parser.add_argument('--model_name', type=str, default= '/mnist6_q8_rl.pth', help='load model')
parser.add_argument('--backend', type=str, default= 'ibmq_kolkata', help='load model')
parser.add_argument('--epath', type=str, default= 'excels/ibmq_kolkata_info1.xlsx', help='load model')



args = parser.parse_args()


train_flag = args.train
n_epochs   = args.epochs
backend_name = args.backend
model_path = args.mpath + backend_name + args.model_name
excel_path = args.epath



if __name__ == '__main__':

    IBMQ.load_account()
    
    provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='sys-reserve')
    

    main(backend_name)
