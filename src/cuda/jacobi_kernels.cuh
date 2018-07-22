#ifndef __JACOBI_KERNELS_CUH__
#define __JACOBI_KERNELS_CUH__

#include "vector_types.h"

__global__ void jacobi
	(
		float* xnew,
		int* row, 
		int* col,
		float* val, 
		float* x, 
		float* b, 
		int n
	);

#endif