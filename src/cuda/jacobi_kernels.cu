#include "jacobi_kernels.cuh"

__global__ void jacobi
(
	float* xnew,
	int* row, 
	int* col,
	float* val, 
	float* x, 
	float* b, 
	int n
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while(index + offset < n)
	{
		float temp = 0.0f;
		float aii = 0.0f;
		for(int i = row[index + offset]; i < row[index + offset + 1]; i++)
		{	
			if(col[i] != index + offset)
			{
				temp += val[i] * x[col[i]];
			}
			else
			{
				aii = val[i];
			}
		}

		xnew[index + offset] = (b[index + offset] - temp) / aii;

		offset += blockDim.x * gridDim.x;
	}
}