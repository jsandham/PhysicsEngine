#ifndef __SOLID_KERNELS_CUH__
#define __SOLID_KERNELS_CUH__

#include "vector_types.h"

namespace SolidKernels
{
	__global__ void compute_local_stiffness_matrices
		(
			float4* pos,
			float* localStiffnessMatrices,
			int* connect,
			int ne,
			int npe,
			int type
		);

	__global__ void compute_local_mass_matrices
		(
			float4* pos,
			float* localMassMatrices,
			int* connect,
			int ne,
			int npe,
			int type
		);
}

#endif