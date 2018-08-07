#include "../../include/cuda/math_kernels.cuh"

// sparse CRS matrix vector product
__global__ void crs_spmv
(
	float* dest,
	int* row, 
	int* col, 
	float* val, 
	float* x,
	int n
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while(index + offset < n){
		float temp = 0.0f;
		for(int i = row[index + offset]; i < row[index + offset + 1]; i++){
			temp += val[i] * x[col[i]];
		}

		dest[index + offset] = temp;

		offset += blockDim.x * gridDim.x;
	}
}

// matrix vector product with a diagonal matrix stored as a single vector
__global__ void diag_spmv
(
	float* dest,
	float* diag,
	float* x,
	int n
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while(index + offset < n){
		dest[index + offset] = diag[index + offset] * x[index + offset];

		offset += blockDim.x * gridDim.x;
	}
}

// single precision dot product product = x * y
__global__ void sdot
(
	float* product,
	float* x,
	float* y,
	int n
)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = blockDim.x*gridDim.x;

	__shared__ float cache[256];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}


	if(threadIdx.x == 0){
		atomicAdd(product, cache[0]);
	}
}

// single precision dest = a*x + y
__global__ void saxpy
(
	float* dest,
	float* x,
	float* y,
	float a,
	int n
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while(index + offset < n){
		dest[index + offset] = y[index + offset] + a * x[index + offset];

		offset += blockDim.x * gridDim.x;
	}
}