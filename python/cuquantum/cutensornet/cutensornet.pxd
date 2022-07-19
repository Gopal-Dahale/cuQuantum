# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# The C types are prefixed with an underscore because we are not
# yet protected by the module namespaces as done in CUDA Python.
# Once we switch over the names would be prettier (in the Cython
# layer).

from libc.stdint cimport int32_t

from cuquantum.utils cimport DataType, DeviceAllocType, DeviceFreeType, Stream


cdef extern from '<cutensornet.h>' nogil:
    # cuTensorNet types
    ctypedef void* _Handle 'cutensornetHandle_t'
    ctypedef int _Status 'cutensornetStatus_t'
    ctypedef void* _NetworkDescriptor 'cutensornetNetworkDescriptor_t'
    ctypedef void* _ContractionPlan 'cutensornetContractionPlan_t'
    ctypedef void* _ContractionOptimizerConfig 'cutensornetContractionOptimizerConfig_t'
    ctypedef void* _ContractionOptimizerInfo 'cutensornetContractionOptimizerInfo_t'
    ctypedef void* _ContractionAutotunePreference 'cutensornetContractionAutotunePreference_t'
    ctypedef void* _WorkspaceDescriptor 'cutensornetWorkspaceDescriptor_t'
    ctypedef void* _SliceGroup 'cutensornetSliceGroup_t'

    # cuTensorNet structs
    ctypedef struct _NodePair 'cutensornetNodePair_t':
        int first
        int second
    ctypedef struct _ContractionPath 'cutensornetContractionPath_t':
        int numContractions
        _NodePair *data
    ctypedef struct _DeviceMemHandler 'cutensornetDeviceMemHandler_t':
        void* ctx
        DeviceAllocType device_alloc
        DeviceFreeType device_free
        # Cython limitation: cannot use C defines in declaring a static array,
        # so we just have to hard-code CUTENSORNET_ALLOCATOR_NAME_LEN here...
        char name[64]
    ctypedef void(*LoggerCallbackData 'cutensornetLoggerCallbackData_t')(
        int32_t logLevel,
        const char* functionName,
        const char* message,
        void* userData)

    # cuTensorNet enums
    ctypedef enum _ComputeType 'cutensornetComputeType_t':
        pass

    ctypedef enum _GraphAlgo 'cutensornetGraphAlgo_t':
        CUTENSORNET_GRAPH_ALGO_RB
        CUTENSORNET_GRAPH_ALGO_KWAY

    ctypedef enum _MemoryModel 'cutensornetMemoryModel_t':
        CUTENSORNET_MEMORY_MODEL_HEURISTIC
        CUTENSORNET_MEMORY_MODEL_CUTENSOR

    ctypedef enum _OptimizerCost 'cutensornetOptimizerCost_t':
        CUTENSORNET_OPTIMIZER_COST_FLOPS
        CUTENSORNET_OPTIMIZER_COST_TIME
        CUTENSORNET_OPTIMIZER_COST_TIME_TUNED

    ctypedef enum _ContractionOptimizerConfigAttribute 'cutensornetContractionOptimizerConfigAttributes_t':
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_PARTITIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_CUTOFF_SIZE
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_ALGORITHM
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_ITERATIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_ITERATIONS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_RECONFIG_NUM_LEAVES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_DISABLE_SLICING
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_MODEL
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MEMORY_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_MIN_SLICES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SLICER_SLICE_FACTOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SIMPLIFICATION_DISABLE_DR
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED
        CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_COST_FUNCTION_OBJECTIVE

    ctypedef enum _ContractionOptimizerInfoAttribute 'cutensornetContractionOptimizerInfoAttributes_t':
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICED_MODES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_MODE
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICED_EXTENT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PHASE1_FLOP_COUNT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_LARGEST_TENSOR
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_SLICING_OVERHEAD
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_INTERMEDIATE_MODES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_INTERMEDIATE_MODES
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_EFFECTIVE_FLOPS_EST
        CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_RUNTIME_EST

    ctypedef enum _ContractionAutotunePreferenceAttribute 'cutensornetContractionAutotunePreferenceAttributes_t':
        CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS
        CUTENSORNET_CONTRACTION_AUTOTUNE_INTERMEDIATE_MODES

    ctypedef enum _WorksizePref 'cutensornetWorksizePref_t':
        CUTENSORNET_WORKSIZE_PREF_MIN
        CUTENSORNET_WORKSIZE_PREF_RECOMMENDED
        CUTENSORNET_WORKSIZE_PREF_MAX

    ctypedef enum _Memspace 'cutensornetMemspace_t':
        CUTENSORNET_MEMSPACE_DEVICE

    # cuTensorNet consts
    int CUTENSORNET_MAJOR
    int CUTENSORNET_MINOR
    int CUTENSORNET_PATCH
    int CUTENSORNET_VERSION
    int CUTENSORNET_ALLOCATOR_NAME_LEN
