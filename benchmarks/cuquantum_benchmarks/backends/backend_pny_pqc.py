# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import os
import time
import warnings

import numpy as np
try:
    import pennylane
except ImportError:
    pennylane = None

# pennylane.enable_return()

from .backend import Backend
from .._utils import is_running_mpi


# set up a logger
logger_name = "cuquantum-benchmarks"
logger = logging.getLogger(logger_name)


class PennylanePQC(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if pennylane is None:
            raise RuntimeError("pennylane is not installed")
        self.dtype = np.complex128
        print("Dtype", self.dtype)
        self.identifier = identifier
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nqubits = kwargs.pop('nqubits')
        self.circuit = None

    def _make_qnode(self, circuit, nshots=0, **kwargs):
        print("N SHOTS", nshots)
        if nshots > 0:
            raise ValueError("Only analytic gradient is supported")

        if self.identifier == "pennylane-lightning-gpu-pqc":
            if self.ngpus:
                try:
                    import pennylane_lightning_gpu
                except ImportError as e:
                    raise RuntimeError("PennyLane-Lightning-GPU plugin is not installed") from e
                if self.ngpus > 1:
                    warnings.warn(f"Only multiple measurements is supported with --ngpus > 1 for the backend {self.identifier}")
            dev = pennylane.device("lightning.gpu", wires=self.nqubits, c_dtype=self.dtype, batch_obs=True)
        elif self.identifier == "pennylane-lightning-kokkos-pqc":
            # there's no way for us to query what execution space (=backend) that kokkos supports at runtime,
            # so let's just set up Kokkos::InitArguments and hope kokkos to do the right thing...
            try:
                import pennylane_lightning_kokkos
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning-Kokkos plugin is not installed") from e
            args = pennylane_lightning_kokkos.lightning_kokkos.InitArguments()
            args.num_threads = self.ncpu_threads
            args.disable_warnings = int(logger.getEffectiveLevel() != logging.DEBUG)
            ## Disable MPI because it's unclear if pennylane actually supports it (at least it's untested)
            # # if we're running MPI, we want to know now and get it init'd before kokkos is
            # MPI = is_running_mpi()
            # if MPI:
            #     comm = MPI.COMM_WORLD
            #     args.ndevices = min(comm.Get_size(), self.ngpus)  # note: kokkos uses 1 GPU per process
            dev = pennylane.device(
                "lightning.kokkos", wires=self.nqubits, c_dtype=self.dtype, batch_obs=True,
                sync=False,
                kokkos_args=args)
        elif self.identifier == "pennylane-lightning-qubit-pqc":
            try:
                import pennylane_lightning
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning plugin is not installed") from e
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.ncpu_threads > 1 and self.ncpu_threads != int(os.environ.get("OMP_NUM_THREADS", "-1")):
                warnings.warn(f"--ncputhreads is ignored, for {self.identifier} please set the env var OMP_NUM_THREADS instead",
                              stacklevel=2)
            dev = pennylane.device("lightning.qubit", wires=self.nqubits, c_dtype=self.dtype, batch_obs=True)
        elif self.identifier == "pennylane-pqc":
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            dev = pennylane.device("default.qubit", wires=self.nqubits, c_dtype=self.dtype)
        else:
            raise ValueError(f"the backend {self.identifier} is not recognized")

        qnode = pennylane.QNode(circuit, device=dev, diff_method='best')
        return qnode

    def preprocess_circuit(self, general_circuit, *args, **kwargs):
        nshots = kwargs.get('nshots', 0)
        print("N SHOTS in PREPROCESS", nshots)
        t1 = time.perf_counter()
        self.circuit = self._make_qnode(general_circuit.circuit, nshots)
        # self.circuit = lambda x: pennylane.numpy.hstack(qnode(x))
        self.params = general_circuit.values
        t2 = time.perf_counter()
        time_make_qnode = t2-t1
        logger.info(f'make qnode took {time_make_qnode} s')
        return {'make_qnode': time_make_qnode}

    def run(self, circuit, nshots=1024):
        # both circuit & nshots are set in preprocess_circuit()
        # jac = pennylane.numpy.hstack(self.circuit(self.params))
        jac = self.circuit(self.params)
        # jac = pennylane.jacobian(self.circuit)(self.params)
        results = jac
        post_res = None # TODO
        run_data = {}
        return {'results': results, 'post_results': post_res, 'run_data': run_data}


PnyLightningGpuPQC = functools.partial(PennylanePQC, identifier='pennylane-lightning-gpu-pqc')
PnyLightningCpuPQC = functools.partial(PennylanePQC, identifier='pennylane-lightning-qubit-pqc')
PnyLightningKokkosPQC = functools.partial(PennylanePQC, identifier='pennylane-lightning-kokkos-pqc')
PnyPQC = functools.partial(PennylanePQC, identifier='pennylane-pqc')
