# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .backend_cirq import Cirq
from .backend_cirq_pqc import CirqPQC
from .backend_cutn import cuTensorNet
from .backend_pny import Pny, PnyLightningGpu, PnyLightningCpu, PnyLightningKokkos
from .backend_pny_pqc import PnyPQC, PnyLightningGpuPQC, PnyLightningCpuPQC, PnyLightningKokkosPQC
from .backend_qsim import Qsim, QsimCuda, QsimCusv, QsimMgpu
from .backend_qsim_pqc import QsimPQC, QsimCudaPQC, QsimCusvPQC, QsimMgpuPQC
from .backend_qiskit import Aer, AerCuda, AerCusv, CusvAer
from .backend_qulacs import QulacsGpu, QulacsCpu
try:
    from .backend_naive import Naive
except ImportError:
    Naive = None


backends = {
    'aer': Aer,
    'aer-cuda': AerCuda,
    'aer-cusv': AerCusv,
    'cusvaer': CusvAer,
    'cirq': Cirq,
    'cirq-pqc': CirqPQC,
    'cutn': cuTensorNet,
    'qsim': Qsim,
    'qsim-cuda': QsimCuda,
    'qsim-cusv': QsimCusv,
    'qsim-mgpu': QsimMgpu,
    'qsim-pqc': QsimPQC,
    'qsim-cuda-pqc': QsimCudaPQC,
    'qsim-cusv-pqc': QsimCusvPQC,
    'qsim-mgpu-pqc': QsimMgpuPQC,
    'pennylane': Pny,
    'pennylane-lightning-gpu': PnyLightningGpu,
    'pennylane-lightning-qubit': PnyLightningCpu,
    'pennylane-lightning-kokkos': PnyLightningKokkos,
    'pennylane-pqc': PnyPQC,
    'pennylane-lightning-gpu-pqc': PnyLightningGpuPQC,
    'pennylane-lightning-qubit-pqc': PnyLightningCpuPQC,
    'pennylane-lightning-kokkos-pqc': PnyLightningKokkosPQC,
    'qulacs-cpu': QulacsCpu,
    'qulacs-gpu': QulacsGpu,
}
if Naive:
    backends['naive'] = Naive


def createBackend(backend_name, ngpus, ncpu_threads, precision, *args, **kwargs):
    return backends[backend_name](ngpus, ncpu_threads, precision, *args, **kwargs)
