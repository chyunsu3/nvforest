#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "cuda_runtime_api.h" nogil:
    ctypedef struct CUstream_st:
        pass
    ctypedef CUstream_st* cudaStream_t


cdef extern from "nvforest/cuda_stream.hpp" namespace "nvforest" nogil:
    ctypedef cudaStream_t cuda_stream
