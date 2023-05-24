# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
try:
    import qsimcirq
except ImportError:
    qsimcirq = None

import cupy as cp
from .backend import Backend
import time
import logging
import tensorflow_quantum as tfq
import tensorflow as tf

# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class QsimCirqPQC(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if qsimcirq is None:
            raise RuntimeError("qsimcirq is not installed")
        if precision != 'single':
            raise ValueError("all qsim backends only support single precision")
        self.identifier = identifier
        qsim_options = self.create_qsim_options(identifier, ngpus, ncpu_threads, **kwargs)
        self.backend = qsimcirq.QSimSimulator(qsim_options=qsim_options)

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

        if self.identifier == "qsim-mgpu-pqc":
            dev = cp.cuda.Device()
        if nshots > 0:
            raise ValueError("Only analytic gradient is supported")
        else:
            with tf.GradientTape() as g:
                g.watch(self.values_tensor)
                ev_list = self.exp_layer(self.circuit, operators=self.obs, symbol_names=self.symbols, symbol_values= self.values_tensor)
            grads = g.gradient(ev_list, self.values_tensor)

        if self.identifier == "qsim-mgpu-pqc":
            # work around a bug
            if dev != cp.cuda.Device():
                dev.use()

        results = (ev_list, grads)
        post_res = results
        return {'results': results, 'post_results': post_res, 'run_data': run_data}
    
    @staticmethod
    def create_qsim_options(identifier, ngpus, ncpu_threads, **kwargs):
        nfused = kwargs.pop('nfused')
        if identifier == "qsim-mgpu-pqc":
            if ngpus >= 1:
                # use cuQuantum Appliance interface
                ops = qsimcirq.QSimOptions(gpu_mode=tuple(range(ngpus)), max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus for the backend {identifier}")
        elif identifier == "qsim-cuda-pqc":
            if ngpus == 1:
                try:
                    # use public interface
                    ops = qsimcirq.QSimOptions(gpu_mode=0, use_gpu=True, max_fused_gate_size=nfused)
                except TypeError:
                    # use cuQuantum Appliance interface
                    ops = qsimcirq.QSimOptions(gpu_mode=0, disable_gpu=False, use_sampler=False, max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus 1 for the backend {identifier}")
        elif identifier == "qsim-cusv-pqc":
            if ngpus == 1:
                try:
                    # use public interface
                    ops = qsimcirq.QSimOptions(gpu_mode=1, use_gpu=True, max_fused_gate_size=nfused)
                except TypeError:
                    # use cuQuantum Appliance interface
                    ops = qsimcirq.QSimOptions(gpu_mode=1, disable_gpu=False, use_sampler=False, max_fused_gate_size=nfused)
            else:
                raise ValueError(f"need to specify --ngpus 1 for the backend {identifier}")
        elif identifier == "qsim-pqc":
            if ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {identifier}")
            try:
                # use public interface
                ops = qsimcirq.QSimOptions(use_gpu=False, cpu_threads=ncpu_threads, max_fused_gate_size=nfused)
            except TypeError:
                # use cuQuantum Appliance interface
                ops = qsimcirq.QSimOptions(disable_gpu=True, use_sampler=False, cpu_threads=ncpu_threads, max_fused_gate_size=nfused,
                                           gpu_mode=0)
        else:
            raise ValueError(f"the backend {identifier} is not recognized")

        return ops

QsimMgpuPQC = functools.partial(QsimCirqPQC, identifier='qsim-mgpu-pqc')
QsimCudaPQC = functools.partial(QsimCirqPQC, identifier='qsim-cuda-pqc')
QsimCusvPQC = functools.partial(QsimCirqPQC, identifier='qsim-cusv-pqc')
QsimPQC = functools.partial(QsimCirqPQC, identifier='qsim-pqc')
