# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
from cmath import pi, exp

try:
    import pennylane
except ImportError:
    pennylane = None

from .frontend import Frontend, GeneralCircuit


class Pennylane(Frontend):

    def __init__(self, nqubits, config):
        if pennylane is None:
            raise RuntimeError('pennylane is not installed')

        self.nqubits = nqubits
        self.config = config

    def generateCircuit(self, gateSeq):
        meas_wires = [g.targets[0] for g in gateSeq if g.id == 'measure']
    
        import pennylane.numpy as pnp

        t_params = [g.params for g in gateSeq if g.symbol] # trainable params
        t_params = pnp.array(t_params, requires_grad=True)

        def circuit_body(params):
            
            c = 0
            for g in gateSeq:
                if g.id =='h': 
                    pennylane.Hadamard(wires=g.targets)

                elif g.id =='x': 
                    pennylane.PauliX(wires=g.targets)

                elif g.id =='cnot': 
                    pennylane.CNOT(wires=[g.controls, g.targets])

                elif g.id =='cz': 
                    pennylane.CZ(wires=[g.controls, g.targets])

                elif g.id =='rz': 
                    if g.symbol:
                        pennylane.RZ(params[c], g.targets)
                        c+=1
                    else:
                        pennylane.RZ(g.params, g.targets)

                elif g.id =='rx': 
                    if g.symbol:
                        pennylane.RX(params[c], g.targets)
                        c+=1
                    else:
                        pennylane.RX(g.params, g.targets)

                elif g.id =='ry': 
                    if g.symbol:
                        pennylane.RY(params[c], g.targets)
                        c+=1
                    else:
                        pennylane.RY(g.params, g.targets)

                elif g.id =='czpowgate': 
                    CZPow_matrix = [[1,0],[0,exp(1j*pi*g.params)]]
                    pennylane.ControlledQubitUnitary(CZPow_matrix,control_wires=g.controls, wires=g.targets)

                elif g.id =='swap': 
                    pennylane.SWAP(wires=[g.targets[0], g.targets[1]])

                elif g.id =='cu': 
                    pennylane.ControlledQubitUnitary(g.matrix, control_wires=g.controls, wires=g.targets)

                elif g.id == 'u':  
                    pennylane.QubitUnitary(g.matrix, wires=g.targets)

                elif g.id == "measure":
                    pass

                else:
                    raise NotImplementedError(f"The gate type {g.id} is not defined")

        if len(meas_wires) == 0:
            def circuit(params):
                circuit_body(params)
                return pennylane.expval(pennylane.PauliZ(0))
                # return [pennylane.expval(pennylane.PauliZ(q)) for q in range(self.nqubits)]
        else:
            def circuit(params):
                circuit_body(params)
                return pennylane.sample(wires=meas_wires)

        return GeneralCircuit(circuit, t_params, None, None)