import torch.nn as nn
import torch
import numpy as np
import pennylane as qml
n_qubits = 3
q_depth = 2


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


dev = qml.device('default.qubit', wires=n_qubits)


@qml.qnode(dev, interface='torch')
def q_net(q_in, q_weights_flat, max_layers):

    # Reshape weights
    q_weights = q_weights_flat.reshape(max_layers, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_in)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k+1])

    # Expectation values in the Z basis
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]


class Quantumnet(nn.Module):
    def __init__(self, n_output, device, max_layers=15, q_delta=0.01):
        super().__init__()
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(
            q_delta * torch.randn(max_layers * n_qubits))
        self.post_net = nn.Linear(n_qubits, n_output)
        self.device = device
        self.max_layers = max_layers

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch, and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = torch.Tensor(
                q_net(elem, self.q_params, self.max_layers)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
        return self.post_net(q_out)
