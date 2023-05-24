# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
import time
import logging

try:
    import cirq
except ImportError:
    cirq = None

from .backend import Backend
import tensorflow_quantum as tfq
import tensorflow as tf

# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)

class CirqPQC(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, **kwargs):
        if cirq is None:
            raise RuntimeError("cirq is not installed")
        if ngpus > 0:
            raise ValueError("the cirq backend only runs on CPU")
        if ncpu_threads > 1:
            warnings.warn("cannot set the number of CPU threads for the cirq backend")
        if precision != 'single':
            raise ValueError("the cirq backend only supports single precision")

        self.backend = cirq.Simulator()
        logger.info("CIRQ PQC BACKEND")

    def preprocess_circuit(self, general_circuit, *args, **kwargs):
        t1 = time.perf_counter()

        # Extract circuit data from general_circuit
        self.circuit = general_circuit.circuit
        self.symbols = general_circuit.symbols
        self.obs = general_circuit.obs
        values = general_circuit.values

        # preprocessing
        adjoint_differentiator = tfq.differentiators.Adjoint()
        self.values_tensor = tf.convert_to_tensor([values])
        self.exp_layer = tfq.layers.Expectation(backend= self.backend, differentiator= adjoint_differentiator)

        t2 = time.perf_counter()
        time_get_circuit = t2-t1
        
        logger.info(f'preprocess circuit took {time_get_circuit} s')
        return {'preprocess_circuit': time_get_circuit}

    def run(self, circuit, nshots=0):
        # circuit is set in preprocess_circuit()
        run_data = {}
        if nshots > 0:
            raise ValueError("Only analytic gradient is supported")
        else:
            with tf.GradientTape() as g:
                g.watch(self.values_tensor)
                ev_list = self.exp_layer(self.circuit, operators=self.obs, symbol_names=self.symbols, symbol_values= self.values_tensor)
            grads = g.gradient(ev_list, self.values_tensor)

        results = (ev_list, grads)
        post_res = results
        return {'results': results, 'post_results': post_res, 'run_data': run_data}
