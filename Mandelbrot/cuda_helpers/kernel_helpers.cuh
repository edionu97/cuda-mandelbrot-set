#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// Wrapper for creating a barrier
/// </summary>
__device__ inline void  set_barrier()
{
	__syncthreads();
}


