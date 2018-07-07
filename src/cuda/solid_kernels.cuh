#ifndef __SOLID_KERNELS_CUH__
#define __SOLID_KERNELS_CUH__

#include "vector_types.h"

// __device__ glm::mat4 jacobian(float xi, float et, float ze);

__global__ void compute_local_stiffness_matrices
	(
		float4* pos,
		float* localStiffnessMatrices,
		int* connect,
		int ne,
		int npe
	);

#endif