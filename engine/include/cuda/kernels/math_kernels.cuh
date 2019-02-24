#ifndef __MATH_KERNELS_H__
#define __MATH_KERNELS_H__

__global__ void crs_spmv
	(
		float* dest,
		int* row, 
		int* col, 
		float* val, 
		float* x,
		int n
	);

__global__ void diag_spmv
 	(
 		float* dest,
 		float* diag,
 		float* x,
 		int n
	);

__global__ void sdot
	(
		float* product,
		float* x,
		float* y,
		int n
	);

__global__ void saxpy
	(
		float* dest,
		float* x,
		float* y,
		float a,
		int n
	);



#endif