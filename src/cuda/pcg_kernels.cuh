#ifndef __PCG_KERNELS_CUH__
#define __PCG_KERNELS_CUH__

#include <vector_types.h>

__global__ void pcg
	(
		int* row, 
		int* col, 
		float* A, 
		float* x, 
		float* b, 
		int n, 
		int maxIter, 
		float tol
	);



#endif