# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import cupy
from cupy.testing import shaped_random
import numpy
try:
    import torch
except ImportError:
    torch = None

from cuquantum import OptimizerOptions
from cuquantum.cutensornet._internal.circuit_converter_utils import EINSUM_SYMBOLS_BASE
from cuquantum.cutensornet._internal.einsum_parser import infer_output_mode_labels

from .data import dtype_names


machine_epsilon_values = [numpy.finfo(dtype).eps for dtype in dtype_names]

rtol_mapper = dict(zip(
    dtype_names,
    [numpy.sqrt(m_eps) for m_eps in machine_epsilon_values]
))

atol_mapper = dict(zip(
    dtype_names,
    [10 * m_eps for m_eps in machine_epsilon_values]
))


def set_path_to_optimizer_options(optimizer_opts, path):
    if optimizer_opts is None:
        optimizer_opts = {"path": path}
    elif isinstance(optimizer_opts, dict):
        optimizer_opts["path"] = path
    else:
        assert isinstance(optimizer_opts, OptimizerOptions)
        optimizer_opts.path = path
    return optimizer_opts


def compute_and_normalize_numpy_path(data, num_operands):
    try:
        # this can fail if the TN is too large (ex: containing unicode)
        path, _ = numpy.einsum_path(*data, optimize=True)
    except:
        raise NotImplementedError
    path = path[1:]

    # now we need to normalize the NumPy path, because NumPy supports
    # contracting a group of tensors at once whereas we only support
    # pairwise contraction
    num_operands -= 1
    norm_path = []
    for indices in path:
        assert all(idx >= 0 for idx in indices)
        if len(indices) >= 2:
            indices = sorted(indices, reverse=True)
            norm_path.append((indices[0], indices[1]))
            num_operands -= 1
            for idx in indices[2:]:
                # keep contracting with the latest intermediate
                norm_path.append((num_operands, idx))
                num_operands -= 1
        else:
            # single TN reduction is supported by NumPy, but we can't handle
            # that, just raise to avoid testing against NumPy path
            assert len(indices) > 0
            raise NotImplementedError

    return norm_path


class EinsumFactory:
    """Take a valid einsum expression and compute shapes, modes, etc for testing."""

    size_dict = dict(zip(EINSUM_SYMBOLS_BASE, (2, 3, 4)*18))

    def __init__(self, einsum_expr):
        self.einsum_expr = einsum_expr
        self.einsum_format = self._decide_format(einsum_expr)
        self._modes = None

    def _decide_format(self, einsum_expr):
        # this is a helper solely for testing purpose
        if isinstance(einsum_expr, str):
            einsum_format = "subscript"
        elif isinstance(einsum_expr, tuple):
            einsum_format = "interleaved"
        else:
            assert False
        return einsum_format

    def _gen_shape(self, modes):
        shape = []

        # find ellipsis, record the position, and remove it
        if isinstance(modes, str):
            ellipsis = modes.find("...")
            if ellipsis >= 0:
                modes = modes.replace("...", "")
        else:
            try:
                ellipsis = modes.index(Ellipsis)
            except ValueError:
                ellipsis = -1
            if ellipsis >= 0:
                modes = modes[:ellipsis] + modes[ellipsis+1:]

        # generate extents for remaining modes
        for mode in modes:
            if mode in self.size_dict:
                extent = self.size_dict[mode]
            else:
                # exotic mode label, let's assign an extent to it
                if isinstance(mode, str):
                    extent = ord(mode) % 3 + 2
                else:
                    extent = abs(hash(mode)) % 3 + 2
                self.size_dict[mode] = extent
            shape.append(extent)

        # put back ellipsis, assuming it has single axis of extent 5
        if ellipsis >= 0:
            shape.insert(ellipsis, 5)

        return shape

    @property
    def input_shapes(self):
        out = []

        for modes in self.input_modes:
            shape = self._gen_shape(modes)
            out.append(shape)

        return out

    @property
    def output_shape(self):
        raise NotImplementedError  # TODO

    @property
    def modes(self):
        if self._modes is None:
            if self.einsum_format == "subscript":
                if "->" in self.einsum_expr:
                    inputs, output = self.einsum_expr.split("->")
                    inputs = inputs.split(",")
                else:
                    inputs = self.einsum_expr
                    inputs = inputs.split(",")
                    output = infer_output_mode_labels(inputs)
            else:
                # output could be a placeholder
                inputs = self.einsum_expr[:-1]
                if self.einsum_expr[-1] is None:
                    output = infer_output_mode_labels(inputs)
                else:
                    output = self.einsum_expr[-1]

            self._modes = tuple(inputs) + tuple([output])
        return self._modes

    @property
    def input_modes(self):
        return self.modes[:-1]

    @property
    def output_modes(self):
        return self.modes[-1]

    def generate_operands(self, shapes, xp, dtype, order):
        # we always generate data from shaped_random as CuPy fixes
        # the RNG seed for us
        if xp == "torch-cpu":
            _xp = numpy
        elif xp == "torch-gpu":
            _xp = cupy
        else:
            _xp = sys.modules[xp]

        operands = [
            shaped_random(shape, xp=_xp, dtype=dtype, order=order)
            for shape in shapes
        ]

        if xp == "torch-cpu":
            operands = [torch.as_tensor(op, device="cpu") for op in operands]
        elif xp == "torch-gpu":
            operands = [torch.as_tensor(op, device="cuda") for op in operands]

        return operands

    def convert_by_format(self, operands, *, dummy=False):
        if dummy:
            # create dummy NumPy arrays to bypass the __array_function__
            # dispatcher, see numpy/numpy#21379 for discussion
            operands = [numpy.broadcast_to(0, arr.shape) for arr in operands]

        if self.einsum_format == "subscript":
            data = [self.einsum_expr, *operands]
        elif self.einsum_format == "interleaved":
            modes = [tuple(modes) for modes in self.input_modes]
            data = [i for pair in zip(operands, modes) for i in pair]
            data.append(tuple(self.output_modes))

        return data
