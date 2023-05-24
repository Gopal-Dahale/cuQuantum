# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import random

from .benchmark import Benchmark
from .._utils import Gate


class Differentiator(Benchmark):

    # TODO: this should be frontend's property
    gate_types = ('h', 'x', 'rz', 'rx', 'ry', 'cnot', 'cz', 'swap')

    @staticmethod
    def generateGatesSequence(nqubits, config):
        try:
            num_layers = config['num_layers']
        except KeyError:
            num_layers = 1

        seed = np.iinfo(np.int32).max
        rng = np.random.default_rng(seed)
        thetas = rng.random((num_layers, nqubits, 3))

        circuit = []

        # apply arbitrary random operations at every layer
        for l in range(num_layers):
            for q in range(nqubits):
                circuit.append(Gate(id='rz', params=thetas[l, q, 0], targets=q, symbol=f'x_{l}_{q}_{0}'))
                circuit.append(Gate(id='ry', params=thetas[l, q, 1], targets=q, symbol=f'x_{l}_{q}_{1}'))
                circuit.append(Gate(id='rz', params=thetas[l, q, 2], targets=q, symbol=f'x_{l}_{q}_{2}'))

            circuit += [Gate(id='cnot', controls=q, targets=q+1) for q in range(nqubits-1)]
            circuit += [Gate(id='cnot', controls=nqubits-1, targets=0)]

        return circuit
