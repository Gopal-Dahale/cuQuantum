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

# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)

class Cirq(Backend):

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

    def preprocess_circuit(self, general_circuit, *args, **kwargs):
        t1 = time.perf_counter()
        self.circuit = general_circuit.circuit
        t2 = time.perf_counter()
        time_get_circuit = t2-t1
        logger.info(f'preprocess circuit took {time_get_circuit} s')
        return {'preprocess_circuit': time_get_circuit}

    def run(self, circuit, nshots=1024):
        # circuit is set in preprocess_circuit()
        run_data = {}
        if nshots > 0:
            results = self.backend.run(self.circuit, repetitions=nshots)
        else:
            results = self.backend.simulate(self.circuit)
        post_res = results.measurements['result']
        return {'results': results, 'post_results': post_res, 'run_data': run_data}
