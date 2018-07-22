#include "CudaSolvers.cuh"

#include "math_kernels.cuh"
#include "jacobi_kernels.cuh"

using namespace PhysicsEngine;

// TODO: write this method
int CudaSolvers::jacobi(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, int maxIter, float tol, CudaJacobi* jacobi)
{
	int iter = 0;
	while(iter < maxIter){
		// dim3 blockSize(64, 1);
		// dim3 gridSize(64, 1);

		// jacobi<<<blockSize, gridSize>>>
		// (
		// 	xnew,
		// 	row, 
		// 	col,
		// 	val, 
		// 	x, 
		// 	b, 
		// 	n
		// );

		// x = xnew;

		iter++;
	}	

	return iter;
}

// TODO: write this method
int CudaSolvers::pcg(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, int maxIter, float tol, CudaPCG* pcg)
{
	int iter = 0;
	while(iter < maxIter){

		iter++;
	}

	return iter;
}

// TODO: write this method
int CudaSolvers::amg(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, float theta, float tol, CudaAMG* amg)
{
	int iter = 0;

	return iter;
}